#ifndef BUFFER_CL_H
#define BUFFER_CL_H

#include "Storage.h"

namespace GPU {
	namespace CL {

	template <typename T>
	class BufferCL {
	public:
		typedef std::shared_ptr<BufferCL> Ptr;

		const cl::Buffer& Get() const { return _buffer; }

		size_t Size() const { return _impl->RawSize(); }

		size_t DeviceBytesOffset() const { return _deviceOffset * sizeof(T); }
		size_t DeviceSizeFromOffset() const { return _impl->RawSize() - (_deviceOffset * sizeof(T)); }

		size_t HostBytesOffset() const { return _hostOffset * sizeof(T); }
		size_t HostSizeFromOffset() const { return _impl->RawSize() - (_hostOffset * sizeof(T)); }

		size_t Count() const { return _impl->Size(); }
		cl_mem_flags Flags() const { return _flags; }
		T* Data() const { return _impl->Data(); }

		BufferCL(const cl::Context& c, typename Storage<T>::Ptr i, const cl_mem_flags& f)
			: _impl(i), _flags(f), _deviceOffset(0), _hostOffset(0) {
			_buffer = cl::Buffer(c, f, i->RawSize(), i->Data());
		}

	private:

		cl::Buffer _buffer;
		typename Storage<T>::Ptr _impl;

		cl_mem_flags _flags;
		size_t _deviceOffset, _hostOffset;
	};

	/*
		Device Origin :	buffer_origin[2] * buffer_slice_pitch + buffer_origin[1] * buffer_row_pitch + buffer_origin[0]
		Host Origin	  :	host_origin[2] * host_slice_pitch + host_origin[1] * host_row_pitch + host_origin[0];
	*/
	template <typename T>
	struct BufferRectCL {
		BufferRectCL(size_t regionW, size_t RegionH);

		void HostOrigin(size_t x, size_t y = 0, size_t z = 0) {
			_HostOrigin[0] = x * sizeof(T);
			_HostOrigin[1] = y * sizeof(T);
			_HostOrigin[2] = z * sizeof(T);
		}

		const cl::size_t<3>& HostOrigin() const { return _HostOrigin; }

		void DeviceOrigin(size_t x, size_t y = 0, size_t z = 0) {
			_DeviceOrigin[0] = x * sizeof(T);
			_DeviceOrigin[1] = y * sizeof(T);
			_DeviceOrigin[2] = z * sizeof(T);
		} 

		const cl::size_t<3>& DeviceOrigin() const { return _DeviceOrigin; }

		void HostPitch(size_t row, size_t slice = 0) {
			HostRow = row * sizeof(T);
			HostSlice = slice * sizeof(T);
		}

		void DevicePitch(size_t row, size_t slice = 0) {
			DeviceRow = row * sizeof(T);
			DeviceSlice = slice * sizeof(T);
		}

		size_t HostRow = 0, HostSlice = 0;
		size_t DeviceRow = 0, DeviceSlice = 0;
		cl::size_t<3> Region;

	private:
		cl::size_t<3> _HostOrigin, _DeviceOrigin;
	};

	template <typename T>
	BufferRectCL<T>::BufferRectCL(
		size_t regionW, size_t RegionH
	) {
		Region[0] = regionW * sizeof(T), Region[1] = RegionH * sizeof(T), Region[2] = 1;
		_HostOrigin[0] = 0, _HostOrigin[1] = 0, _HostOrigin[2] = 0;
		_DeviceOrigin[0] = 0, _DeviceOrigin[1] = 0, _DeviceOrigin[2] = 0;
 	}
}}
#endif