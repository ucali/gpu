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
			cl_uint DeviceVendorId;
			cl_uint BaseAddressAlign;
			size_t MaxBufferSize;

			std::string Vendor;
		};

		DeviceCL(const cl::Context& c, const cl::Device& d) {
			_device = d;

			_queue.reset(new QueueCL(c, d));
			_info.reset(new Info);

			_info->Type = d.getInfo<CL_DEVICE_TYPE>();
			_info->BaseAddressAlign = d.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
			_info->DeviceVendorId = d.getInfo<CL_DEVICE_VENDOR_ID>();
			_info->MaxComputeUnit = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
			_info->MaxBufferSize = d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();

			_info->Vendor = std::string(d.getInfo<CL_DEVICE_VENDOR>());
		}

		const cl::Device& Get() const { return _device; }
		cl::Device& Get() { return _device; }

		const Info& GetInfo() const { return *_info; }

		QueueCL& Queue() { return *_queue; }

		bool IsDefault() const {
			cl::Device dev = cl::Device::getDefault();
			return GetInfo().DeviceVendorId == dev.getInfo<CL_DEVICE_VENDOR_ID>();
		}

	private:
		cl::Device _device;

		std::shared_ptr<Info> _info;
		QueueCL::Ptr _queue;
	};

}}

#endif