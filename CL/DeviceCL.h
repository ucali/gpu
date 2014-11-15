#ifndef DEVICE_CL_H 
#define DEVICE_CL_H

#include "CommonCL.h"
#include "QueueCL.h"

namespace GPU {
namespace CL {

	class DeviceCL {
	public:
		typedef std::shared_ptr<DeviceCL> Ptr;
		typedef std::unique_ptr<DeviceCL> UPtr;

		struct Info {
			cl_device_type Type;
			cl_uint MaxComputeUnit;
			cl_uint VendorDeviceId;
			cl_uint BaseAddressAlign;
			size_t MaxBufferSize;

			std::string Vendor;
		};

		DeviceCL(const cl::Context& c, const cl::Device& d) {
			_device = d;

			_queue.reset(new QueueCL(c, d));

			_info.Type = d.getInfo<CL_DEVICE_TYPE>();
			_info.BaseAddressAlign = d.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
			_info.Vendor = std::string(d.getInfo<CL_DEVICE_VENDOR>());
			_info.VendorDeviceId = d.getInfo<CL_DEVICE_VENDOR_ID>();
			_info.MaxComputeUnit = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
			_info.MaxBufferSize = d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
		}

		const cl::Device& Get() const { return _device; }
		cl::Device& Get() { return _device; }

		const Info& GetInfo() const { return _info; }
		Info& GetInfo() { return _info; }

		QueueCL& Queue() { return *_queue; }

		bool IsDefault() const {
			cl::Device dev = cl::Device::getDefault();
			return GetInfo().VendorDeviceId == dev.getInfo<CL_DEVICE_VENDOR_ID>();
		}

	private:
		cl::Device _device;
		Info _info;

		QueueCL::Ptr _queue;

		U_DISABLE_COPY_AND_ASSIGNMENT(DeviceCL);
	};

}}

#endif