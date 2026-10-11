#pragma once

#ifdef STK_USE_WEBGPU

#include "dim3.h"
#include "types.h"

#include <stk/math/float3.h>
#include <stk/math/matrix3x3f.h>
#include <stk/webgpu/webgpu.h>

namespace stk
{
    class Volume;

    // Volume residing on a WebGPU device.
    //
    // Voxels are stored contiguously in a storage buffer, with x as the
    // fastest varying dimension, i.e. the voxel at (x, y, z) has the index
    //  x + size.x * (y + size.y * z)
    //
    // Supported voxel types are float, float2, float4 and uint8_t. WGSL has
    // no 8-bit types, so kernels have to read uint8_t volumes as an array of
    // u32, with four voxels packed in each element.
    class WebGpuVolume
    {
    public:
        WebGpuVolume();
        // @remark This does not copy the data, use clone if you want a separate copy.
        WebGpuVolume(const WebGpuVolume& other) = default;
        // @remark This does not copy the data, use clone if you want a separate copy.
        WebGpuVolume& operator=(const WebGpuVolume& other) = default;

        // Allocates a zero-initialized volume
        WebGpuVolume(const dim3& size, Type voxel_type);

        // Allocates a volume and uploads the given volume
        WebGpuVolume(const Volume& vol);
        ~WebGpuVolume();

        void allocate(const dim3& size, Type voxel_type);
        void release();

        // Clones the volume, including the data
        WebGpuVolume clone() const;

        // Copies the data and meta data of a volume of matching size and type
        void copy_from(const WebGpuVolume& other);

        bool valid() const;

        dim3 size() const;
        Type voxel_type() const;

        void set_origin(const float3& origin);
        void set_spacing(const float3& spacing);
        void set_direction(const Matrix3x3f& direction);

        const float3& origin() const;
        const float3& spacing() const;
        const Matrix3x3f& direction() const;
        const Matrix3x3f& inverse_direction() const;

        void copy_meta_from(const WebGpuVolume& other);

        // Uploads a volume of matching size and type, including meta data
        void upload(const Volume& vol);

        // Downloads the volume to a new volume
        Volume download() const;

        const webgpu::Buffer& buffer() const;

        // Returns true if the type is supported
        static bool supports_type(Type voxel_type);

    private:
        webgpu::Buffer _buffer;

        dim3 _size;
        Type _voxel_type;

        float3 _origin;
        float3 _spacing;
        Matrix3x3f _direction;
        Matrix3x3f _inverse_direction;
    };
}

#endif // STK_USE_WEBGPU
