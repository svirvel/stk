#pragma once

#ifdef STK_USE_WEBGPU

#include <stk/image/dim3.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Minimal layer on top of WebGPU for running compute kernels on volumes.
//
// All data is kept in storage buffers and kernels are written in WGSL. The
// device is created on first use and lives for the remainder of the process.
// None of this is thread-safe; all calls should be made from the same thread.

// Workgroup size expected by Kernel, to be used in the shader source as
//  @compute @workgroup_size(STK_WEBGPU_WORKGROUP_SIZE)
#define STK_WEBGPU_WORKGROUP_SIZE "8, 8, 4"

namespace stk { namespace webgpu {

    // Initializes WebGPU, this is done implicitly whenever required.
    // Throws a fatal error if no suitable device was found.
    void initialize();

    // Returns true if a device is available, initializing if necessary.
    bool available();

    // Description of the device in use
    std::string device_name();

    // Largest buffer that can be allocated and bound to a kernel, in bytes
    uint64_t max_buffer_size();

    // Blocks until all submitted work has completed
    void wait();

    // Throws a fatal error if the device has reported any errors
    void check_errors();

    // Reference to a buffer on the device
    class Buffer
    {
    public:
        Buffer();
        // Allocates a zero-initialized buffer, size is rounded up to a
        // multiple of 4 bytes.
        explicit Buffer(uint64_t size);
        ~Buffer();

        bool valid() const;
        uint64_t size() const;

        void upload(const void* data, uint64_t size, uint64_t offset = 0);
        void download(void* data, uint64_t size, uint64_t offset = 0) const;

        // Copies the first `size` bytes to host memory and returns a pointer
        // to them. The pointer is valid until unmap() is called.
        const void* map_read(uint64_t size) const;
        void unmap() const;

        // Sets the first `size` bytes to zero (whole buffer if size is 0)
        void clear(uint64_t size = 0);

        // Copies the content of another buffer of the same size
        void copy_from(const Buffer& other);

        // A small read-only buffer, to bind in place of optional inputs
        static const Buffer& dummy();

        struct Impl;
        const std::shared_ptr<Impl>& impl() const { return _impl; }

    private:
        std::shared_ptr<Impl> _impl;
    };

    enum Access
    {
        Access_Read,
        Access_ReadWrite
    };

    // A compute kernel.
    //
    // The shader is expected to have the entry point `main`, a uniform block
    // at binding 0 and storage buffers at binding 1 and onwards (all in
    // group 0), and to use the workgroup size STK_WEBGPU_WORKGROUP_SIZE.
    class Kernel
    {
    public:
        // bindings     : Access for each of the storage buffers
        // params_size  : Size of the uniform block in bytes
        Kernel(
            const std::string& source,
            const std::vector<Access>& bindings,
            size_t params_size
        );
        ~Kernel();

        // Runs the kernel for all invocations within dims. Does not wait for
        // the kernel to complete.
        void run(
            const void* params,
            const std::vector<Buffer>& buffers,
            const dim3& dims
        );

    private:
        struct Impl;
        std::shared_ptr<Impl> _impl;
    };

}} // namespace stk::webgpu

#endif // STK_USE_WEBGPU
