#ifdef STK_USE_WEBGPU

#include "webgpu_volume.h"
#include "volume.h"

#include <stk/common/assert.h>

namespace stk
{

WebGpuVolume::WebGpuVolume() :
    _size{0, 0, 0},
    _voxel_type(Type_Unknown),
    _origin{0, 0, 0},
    _spacing{1, 1, 1},
    _direction(Matrix3x3f::Identity),
    _inverse_direction(Matrix3x3f::Identity)
{
}
WebGpuVolume::WebGpuVolume(const dim3& size, Type voxel_type) : WebGpuVolume()
{
    allocate(size, voxel_type);
}
WebGpuVolume::WebGpuVolume(const Volume& vol) : WebGpuVolume()
{
    allocate(vol.size(), vol.voxel_type());
    upload(vol);
}
WebGpuVolume::~WebGpuVolume()
{
}
void WebGpuVolume::allocate(const dim3& size, Type voxel_type)
{
    ASSERT(size.x > 0 && size.y > 0 && size.z > 0);
    FATAL_IF(!supports_type(voxel_type))
        << "Unsupported voxel type for WebGPU: " << as_string(voxel_type);

    _size = size;
    _voxel_type = voxel_type;
    _buffer = webgpu::Buffer(
        uint64_t(size.x) * size.y * size.z * type_size(voxel_type));
}
void WebGpuVolume::release()
{
    _buffer = webgpu::Buffer();
    _size = {0, 0, 0};
    _voxel_type = Type_Unknown;
}
WebGpuVolume WebGpuVolume::clone() const
{
    ASSERT(valid());
    WebGpuVolume out(_size, _voxel_type);
    out.copy_from(*this);
    return out;
}
void WebGpuVolume::copy_from(const WebGpuVolume& other)
{
    ASSERT(valid() && other.valid());
    ASSERT(_size == other._size);
    ASSERT(_voxel_type == other._voxel_type);

    _buffer.copy_from(other._buffer);
    copy_meta_from(other);
}
bool WebGpuVolume::valid() const
{
    return _buffer.valid();
}
dim3 WebGpuVolume::size() const
{
    return _size;
}
Type WebGpuVolume::voxel_type() const
{
    return _voxel_type;
}
void WebGpuVolume::set_origin(const float3& origin)
{
    _origin = origin;
}
void WebGpuVolume::set_spacing(const float3& spacing)
{
    _spacing = spacing;
}
void WebGpuVolume::set_direction(const Matrix3x3f& direction)
{
    _direction = direction;
    _inverse_direction = _direction.inverse();
}
const float3& WebGpuVolume::origin() const
{
    return _origin;
}
const float3& WebGpuVolume::spacing() const
{
    return _spacing;
}
const Matrix3x3f& WebGpuVolume::direction() const
{
    return _direction;
}
const Matrix3x3f& WebGpuVolume::inverse_direction() const
{
    return _inverse_direction;
}
void WebGpuVolume::copy_meta_from(const WebGpuVolume& other)
{
    _origin = other._origin;
    _spacing = other._spacing;
    _direction = other._direction;
    _inverse_direction = other._inverse_direction;
}
void WebGpuVolume::upload(const Volume& vol)
{
    ASSERT(valid() && vol.valid());
    ASSERT(vol.size() == _size);
    ASSERT(vol.voxel_type() == _voxel_type);

    // Sub-volumes are not necessarily contiguous
    const Volume src = vol.is_contiguous() ? vol : vol.clone();
    _buffer.upload(
        src.ptr(), uint64_t(_size.x) * _size.y * _size.z * type_size(_voxel_type));

    _origin = vol.origin();
    _spacing = vol.spacing();
    set_direction(vol.direction());
}
Volume WebGpuVolume::download() const
{
    ASSERT(valid());

    Volume out(_size, _voxel_type);
    _buffer.download(
        out.ptr(), uint64_t(_size.x) * _size.y * _size.z * type_size(_voxel_type));

    out.set_origin(_origin);
    out.set_spacing(_spacing);
    out.set_direction(_direction);
    return out;
}
const webgpu::Buffer& WebGpuVolume::buffer() const
{
    return _buffer;
}
bool WebGpuVolume::supports_type(Type voxel_type)
{
    return voxel_type == Type_Float
        || voxel_type == Type_Float2
        || voxel_type == Type_Float4
        || voxel_type == Type_UChar;
}

} // namespace stk

#endif // STK_USE_WEBGPU
