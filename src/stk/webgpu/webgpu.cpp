#ifdef STK_USE_WEBGPU

#include "webgpu.h"

#include <stk/common/assert.h>
#include <stk/common/error.h>
#include <stk/common/log.h>

#include <webgpu.h>
#include <wgpu.h>

#include <algorithm>
#include <cstring>

namespace {
    struct Context
    {
        bool initialized = false;
        bool failed = false;

        WGPUInstance instance = nullptr;
        WGPUAdapter adapter = nullptr;
        WGPUDevice device = nullptr;
        WGPUQueue queue = nullptr;
        WGPULimits limits = WGPU_LIMITS_INIT;

        std::string name;
        std::string error;
        int num_errors = 0;
    };

    Context _ctx;

    WGPUStringView string_view(const char* s)
    {
        return WGPUStringView{s, WGPU_STRLEN};
    }

    std::string to_string(WGPUStringView s)
    {
        if (!s.data) return "";
        return s.length == WGPU_STRLEN ? std::string(s.data) : std::string(s.data, s.length);
    }

    void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                    WGPUStringView message, void*, void*)
    {
        if (status == WGPURequestAdapterStatus_Success) {
            _ctx.adapter = adapter;
        }
        else {
            _ctx.error = to_string(message);
        }
    }

    void on_device(WGPURequestDeviceStatus status, WGPUDevice device,
                   WGPUStringView message, void*, void*)
    {
        if (status == WGPURequestDeviceStatus_Success) {
            _ctx.device = device;
        }
        else {
            _ctx.error = to_string(message);
        }
    }

    void on_error(WGPUDevice const*, WGPUErrorType, WGPUStringView message, void*, void*)
    {
        // Only keep the first error, any following are likely a consequence
        if (_ctx.num_errors++ == 0) {
            _ctx.error = to_string(message);
        }
    }

    void on_map(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata, void*)
    {
        if (status != WGPUMapAsyncStatus_Success && _ctx.num_errors++ == 0) {
            _ctx.error = "Failed to map buffer: " + to_string(message);
        }
        *static_cast<bool*>(userdata) = true;
    }

    bool try_initialize()
    {
        if (_ctx.initialized || _ctx.failed) {
            return _ctx.initialized;
        }
        _ctx.failed = true; // Until proven otherwise

        WGPUInstanceDescriptor instance_desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
        _ctx.instance = wgpuCreateInstance(&instance_desc);
        if (!_ctx.instance) {
            _ctx.error = "Failed to create instance";
            return false;
        }

        WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
        options.powerPreference = WGPUPowerPreference_HighPerformance;

        WGPURequestAdapterCallbackInfo adapter_cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
        adapter_cb.mode = WGPUCallbackMode_AllowSpontaneous;
        adapter_cb.callback = on_adapter;
        wgpuInstanceRequestAdapter(_ctx.instance, &options, adapter_cb);
        if (!_ctx.adapter) {
            return false;
        }

        // The default limits are too restrictive (e.g. buffers of 256 MB), so
        // we request everything the adapter supports.
        wgpuAdapterGetLimits(_ctx.adapter, &_ctx.limits);

        WGPUDeviceDescriptor device_desc = WGPU_DEVICE_DESCRIPTOR_INIT;
        device_desc.requiredLimits = &_ctx.limits;
        device_desc.uncapturedErrorCallbackInfo.callback = on_error;

        WGPURequestDeviceCallbackInfo device_cb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
        device_cb.mode = WGPUCallbackMode_AllowSpontaneous;
        device_cb.callback = on_device;
        wgpuAdapterRequestDevice(_ctx.adapter, &device_desc, device_cb);
        if (!_ctx.device) {
            return false;
        }
        _ctx.queue = wgpuDeviceGetQueue(_ctx.device);

        WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
        wgpuAdapterGetInfo(_ctx.adapter, &info);
        _ctx.name = to_string(info.device);
        const std::string description = to_string(info.description);
        if (!description.empty()) {
            _ctx.name += " (" + description + ")";
        }
        wgpuAdapterInfoFreeMembers(info);

        _ctx.error.clear();
        _ctx.failed = false;
        _ctx.initialized = true;

        LOG(Info) << "WebGPU device: " << _ctx.name;
        return true;
    }

    void submit(WGPUCommandEncoder encoder)
    {
        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(_ctx.queue, 1, &commands);
        wgpuCommandBufferRelease(commands);
        wgpuCommandEncoderRelease(encoder);
    }
}

namespace stk { namespace webgpu {

struct Buffer::Impl
{
    ~Impl()
    {
        if (staging) wgpuBufferRelease(staging);
        if (buffer) wgpuBufferRelease(buffer);
    }

    WGPUBuffer buffer = nullptr;
    // For downloads, allocated on demand
    WGPUBuffer staging = nullptr;
    uint64_t size = 0;
};

struct Kernel::Impl
{
    ~Impl()
    {
        if (params) wgpuBufferRelease(params);
        if (pipeline) wgpuComputePipelineRelease(pipeline);
        if (layout) wgpuBindGroupLayoutRelease(layout);
    }

    WGPUBindGroupLayout layout = nullptr;
    WGPUComputePipeline pipeline = nullptr;
    WGPUBuffer params = nullptr;
    // Size of the uniform block and the size passed by the user
    size_t params_size = 0;
    size_t user_params_size = 0;
    size_t num_bindings = 0;
};

void initialize()
{
    if (!try_initialize()) {
        FATAL() << "Failed to initialize WebGPU: "
                << (_ctx.error.empty() ? "no suitable device found" : _ctx.error);
    }
}
bool available()
{
    return try_initialize();
}
std::string device_name()
{
    initialize();
    return _ctx.name;
}
uint64_t max_buffer_size()
{
    initialize();
    return std::min(_ctx.limits.maxBufferSize, _ctx.limits.maxStorageBufferBindingSize);
}
void wait()
{
    initialize();
    wgpuDevicePoll(_ctx.device, true, nullptr);
}
void check_errors()
{
    if (_ctx.num_errors > 0) {
        // Reset to allow the user to recover
        std::string error = _ctx.error;
        _ctx.num_errors = 0;
        _ctx.error.clear();
        FATAL() << "WebGPU error: " << error;
    }
}

Buffer::Buffer()
{
}
Buffer::Buffer(uint64_t size)
{
    initialize();

    size = std::max<uint64_t>((size + 3) & ~uint64_t(3), 4);
    FATAL_IF(size > max_buffer_size())
        << "Buffer of " << size / (1024 * 1024) << " MB exceeds the limit of the "
        << "device (" << max_buffer_size() / (1024 * 1024) << " MB)";

    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    desc.size = size;

    _impl = std::make_shared<Impl>();
    _impl->buffer = wgpuDeviceCreateBuffer(_ctx.device, &desc);
    _impl->size = size;

    FATAL_IF(!_impl->buffer) << "Failed to allocate buffer";
}
Buffer::~Buffer()
{
}
bool Buffer::valid() const
{
    return _impl != nullptr;
}
uint64_t Buffer::size() const
{
    return _impl ? _impl->size : 0;
}
void Buffer::upload(const void* data, uint64_t size, uint64_t offset)
{
    ASSERT(valid());
    ASSERT(offset % 4 == 0);
    ASSERT(offset + size <= _impl->size);

    // Sizes are required to be a multiple of 4
    const uint64_t aligned_size = size & ~uint64_t(3);
    if (aligned_size > 0) {
        wgpuQueueWriteBuffer(_ctx.queue, _impl->buffer, offset, data, aligned_size);
    }
    if (aligned_size != size) {
        uint8_t tail[4] = {0, 0, 0, 0};
        memcpy(tail, static_cast<const uint8_t*>(data) + aligned_size, size - aligned_size);
        wgpuQueueWriteBuffer(_ctx.queue, _impl->buffer, offset + aligned_size, tail, 4);
    }
}
void Buffer::download(void* data, uint64_t size, uint64_t offset) const
{
    ASSERT(offset + size <= _impl->size);
    const uint8_t* src = static_cast<const uint8_t*>(map_read(offset + size));
    memcpy(data, src + offset, size);
    unmap();
}
const void* Buffer::map_read(uint64_t size) const
{
    ASSERT(valid());
    ASSERT(size <= _impl->size);
    size = (size + 3) & ~uint64_t(3);

    if (!_impl->staging) {
        WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        desc.size = _impl->size;
        _impl->staging = wgpuDeviceCreateBuffer(_ctx.device, &desc);
        FATAL_IF(!_impl->staging) << "Failed to allocate buffer";
    }

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(_ctx.device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, _impl->buffer, 0, _impl->staging, 0, size);
    submit(encoder);

    bool done = false;
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = on_map;
    cb.userdata1 = &done;
    wgpuBufferMapAsync(_impl->staging, WGPUMapMode_Read, 0, size, cb);
    while (!done) {
        wgpuDevicePoll(_ctx.device, true, nullptr);
    }
    check_errors();

    const void* ptr = wgpuBufferGetConstMappedRange(_impl->staging, 0, size);
    FATAL_IF(!ptr) << "Failed to map buffer";
    return ptr;
}
void Buffer::unmap() const
{
    ASSERT(valid() && _impl->staging);
    wgpuBufferUnmap(_impl->staging);
}
void Buffer::clear(uint64_t size)
{
    ASSERT(valid());
    ASSERT(size <= _impl->size);
    size = size == 0 ? _impl->size : ((size + 3) & ~uint64_t(3));

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(_ctx.device, nullptr);
    wgpuCommandEncoderClearBuffer(encoder, _impl->buffer, 0, size);
    submit(encoder);
}
void Buffer::copy_from(const Buffer& other)
{
    ASSERT(valid() && other.valid());
    ASSERT(size() == other.size());

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(_ctx.device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(
        encoder, other._impl->buffer, 0, _impl->buffer, 0, _impl->size);
    submit(encoder);
}
const Buffer& Buffer::dummy()
{
    static Buffer buffer(16);
    return buffer;
}

Kernel::Kernel(
    const std::string& source,
    const std::vector<Access>& bindings,
    size_t params_size
)
{
    initialize();

    _impl = std::make_shared<Impl>();
    _impl->params_size = (params_size + 15) & ~size_t(15);
    _impl->user_params_size = params_size;
    _impl->num_bindings = bindings.size();

    std::vector<WGPUBindGroupLayoutEntry> entries(bindings.size() + 1);
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entries[i].binding = uint32_t(i);
        entries[i].visibility = WGPUShaderStage_Compute;
        if (i == 0) {
            entries[i].buffer.type = WGPUBufferBindingType_Uniform;
        }
        else {
            entries[i].buffer.type = bindings[i-1] == Access_Read
                ? WGPUBufferBindingType_ReadOnlyStorage
                : WGPUBufferBindingType_Storage;
        }
    }

    WGPUBindGroupLayoutDescriptor layout_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    layout_desc.entryCount = entries.size();
    layout_desc.entries = entries.data();
    _impl->layout = wgpuDeviceCreateBindGroupLayout(_ctx.device, &layout_desc);

    WGPUPipelineLayoutDescriptor pipeline_layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipeline_layout_desc.bindGroupLayoutCount = 1;
    pipeline_layout_desc.bindGroupLayouts = &_impl->layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
        _ctx.device, &pipeline_layout_desc);

    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl.code = string_view(source.c_str());
    WGPUShaderModuleDescriptor module_desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    module_desc.nextInChain = &wgsl.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(_ctx.device, &module_desc);

    WGPUComputePipelineDescriptor pipeline_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipeline_desc.layout = pipeline_layout;
    pipeline_desc.compute.module = module;
    pipeline_desc.compute.entryPoint = string_view("main");
    _impl->pipeline = wgpuDeviceCreateComputePipeline(_ctx.device, &pipeline_desc);

    if (module) wgpuShaderModuleRelease(module);
    wgpuPipelineLayoutRelease(pipeline_layout);

    WGPUBufferDescriptor params_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    params_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    params_desc.size = _impl->params_size;
    _impl->params = wgpuDeviceCreateBuffer(_ctx.device, &params_desc);

    check_errors();
    FATAL_IF(!_impl->pipeline) << "Failed to create kernel";
}
Kernel::~Kernel()
{
}
void Kernel::run(
    const void* params,
    const std::vector<Buffer>& buffers,
    const dim3& dims
)
{
    ASSERT(buffers.size() == _impl->num_bindings);

    std::vector<uint8_t> padded_params(_impl->params_size, 0);
    memcpy(padded_params.data(), params, _impl->user_params_size);
    wgpuQueueWriteBuffer(_ctx.queue, _impl->params, 0, padded_params.data(),
                         padded_params.size());

    std::vector<WGPUBindGroupEntry> entries(buffers.size() + 1);
    entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
    entries[0].binding = 0;
    entries[0].buffer = _impl->params;
    entries[0].size = _impl->params_size;
    for (size_t i = 0; i < buffers.size(); ++i) {
        ASSERT(buffers[i].valid());
        entries[i+1] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[i+1].binding = uint32_t(i + 1);
        entries[i+1].buffer = buffers[i].impl()->buffer;
        entries[i+1].size = buffers[i].size();
    }

    WGPUBindGroupDescriptor bind_group_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bind_group_desc.layout = _impl->layout;
    bind_group_desc.entryCount = entries.size();
    bind_group_desc.entries = entries.data();
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(_ctx.device, &bind_group_desc);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(_ctx.device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, _impl->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
    // Has to match STK_WEBGPU_WORKGROUP_SIZE
    wgpuComputePassEncoderDispatchWorkgroups(
        pass, (dims.x + 7) / 8, (dims.y + 7) / 8, (dims.z + 3) / 4);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    submit(encoder);

    wgpuBindGroupRelease(bind_group);
}

}} // namespace stk::webgpu

#endif // STK_USE_WEBGPU
