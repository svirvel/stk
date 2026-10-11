#include "catch.hpp"

#include <stk/image/volume.h>
#include <stk/image/webgpu_volume.h>
#include <stk/webgpu/webgpu.h>

#include <string>

using namespace stk;

namespace {
    // out[i] = scale * in[i] + offset
    const char* SCALE_KERNEL = R"(
struct Params {
    dims: vec4<i32>,
    scale: f32,
    offset: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size()" STK_WEBGPU_WORKGROUP_SIZE R"()
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec3<i32>(gid);
    if (any(p >= params.dims.xyz)) {
        return;
    }
    let i = p.x + params.dims.x * (p.y + params.dims.y * p.z);
    dst[i] = params.scale * src[i] + params.offset;
}
)";

    struct ScaleParams
    {
        int32_t dims[4];
        float scale;
        float offset;
    };
}

TEST_CASE("webgpu_volume", "[webgpu]")
{
    if (!webgpu::available()) {
        WARN("No WebGPU device available, skipping");
        return;
    }

    SECTION("float")
    {
        // Odd sizes on purpose
        dim3 dims{7, 5, 3};
        VolumeFloat vol(dims);
        for (int i = 0; i < int(dims.x * dims.y * dims.z); ++i) {
            static_cast<float*>(vol.ptr())[i] = float(i) * 0.5f;
        }
        vol.set_origin({1, 2, 3});
        vol.set_spacing({4, 5, 6});

        WebGpuVolume gpu_vol(vol);
        REQUIRE(gpu_vol.valid());
        REQUIRE(gpu_vol.size() == dims);
        REQUIRE(gpu_vol.voxel_type() == Type_Float);
        REQUIRE(gpu_vol.spacing().y == Approx(5.0f));

        VolumeFloat out = gpu_vol.download();
        REQUIRE(out.origin().z == Approx(3.0f));
        for (int i = 0; i < int(dims.x * dims.y * dims.z); ++i) {
            REQUIRE(static_cast<float*>(out.ptr())[i] == float(i) * 0.5f);
        }
    }
    SECTION("uchar")
    {
        // Size is not a multiple of 4 bytes
        dim3 dims{3, 3, 3};
        VolumeUChar vol(dims);
        for (int i = 0; i < 27; ++i) {
            static_cast<uint8_t*>(vol.ptr())[i] = uint8_t(i * 9);
        }

        VolumeUChar out = WebGpuVolume(vol).download();
        for (int i = 0; i < 27; ++i) {
            REQUIRE(static_cast<uint8_t*>(out.ptr())[i] == uint8_t(i * 9));
        }
    }
    SECTION("float4")
    {
        dim3 dims{4, 4, 4};
        VolumeFloat4 vol(dims, float4{1, 2, 3, 4});
        VolumeFloat4 out = WebGpuVolume(vol).download();
        REQUIRE(out(3, 2, 1).x == 1.0f);
        REQUIRE(out(3, 2, 1).w == 4.0f);
    }
    SECTION("clone")
    {
        dim3 dims{4, 4, 4};
        VolumeFloat vol(dims, 3.0f);
        WebGpuVolume a(vol);
        WebGpuVolume b = a.clone();

        // Modifying the original should not affect the clone
        a.upload(VolumeFloat(dims, 5.0f));

        VolumeFloat out_a = a.download();
        VolumeFloat out_b = b.download();
        REQUIRE(out_a(1, 1, 1) == 5.0f);
        REQUIRE(out_b(1, 1, 1) == 3.0f);
    }
    SECTION("zero_initialized")
    {
        VolumeFloat out = WebGpuVolume(dim3{8, 8, 8}, Type_Float).download();
        REQUIRE(out(7, 7, 7) == 0.0f);
    }
}

TEST_CASE("webgpu_kernel", "[webgpu]")
{
    if (!webgpu::available()) {
        WARN("No WebGPU device available, skipping");
        return;
    }

    dim3 dims{19, 13, 7};
    VolumeFloat vol(dims);
    for (int i = 0; i < int(dims.x * dims.y * dims.z); ++i) {
        static_cast<float*>(vol.ptr())[i] = float(i);
    }

    WebGpuVolume src(vol);
    WebGpuVolume dst(dims, Type_Float);

    webgpu::Kernel kernel(
        SCALE_KERNEL,
        {webgpu::Access_Read, webgpu::Access_ReadWrite},
        sizeof(ScaleParams)
    );

    ScaleParams params = {{int(dims.x), int(dims.y), int(dims.z), 0}, 2.0f, 1.0f};
    kernel.run(&params, {src.buffer(), dst.buffer()}, dims);

    // Parameters should be captured at the time of the call
    WebGpuVolume dst2(dims, Type_Float);
    params.scale = 3.0f;
    kernel.run(&params, {src.buffer(), dst2.buffer()}, dims);

    VolumeFloat out = dst.download();
    VolumeFloat out2 = dst2.download();
    webgpu::check_errors();

    for (int i = 0; i < int(dims.x * dims.y * dims.z); ++i) {
        REQUIRE(static_cast<float*>(out.ptr())[i] == 2.0f * i + 1.0f);
        REQUIRE(static_cast<float*>(out2.ptr())[i] == 3.0f * i + 1.0f);
    }
}
