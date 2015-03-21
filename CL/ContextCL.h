#ifndef CONTEXT_CL_H 
#define CONTEXT_CL_H

#include "DeviceCL.h"
#include "BufferCL.h"
#include "ProgramCL.h"

namespace GPU {
namespace CL {

	class ContextCL {
	public:
		typedef std::shared_ptr<ContextCL> Ptr;
		typedef std::unique_ptr<ContextCL> UPtr;

		ContextCL() : _context(nullptr) {
			std::vector<cl::Device> c;
			c.push_back(cl::Device::getDefault());
			_context.reset(new cl::Context(c));

            _device.emplace_back(new DeviceCL(*_context, cl::Device::getDefault()));
		}

		ContextCL(const std::vector<cl::Device> & d) : _context(nullptr) {
			if (!d.size()) {
				throw std::runtime_error("No device");
			}

			_context.reset(new cl::Context(d));

            for (const auto& dev : d) {
                _device.emplace_back(new DeviceCL(*_context, dev));
			}
		}

		const cl::Context& Get() const { return *_context; }

        const DeviceCL& Device() const { return *_device.at(0); }
        DeviceCL& Device() { return *_device.at(0); }

        const std::vector<DeviceCL::Ptr>& Devices() const { return _device; }
        std::vector<DeviceCL::Ptr>& DeviceList() { return _device; }

		std::vector<DeviceCL::Ptr> GPUs();
		const DeviceCL& GPU();
		std::vector<DeviceCL::Ptr> CPUs();
		const DeviceCL& CPU();
		
		template <typename T>
        typename BufferCL<T>::Ptr NewBuffer(typename Storage<T>::Ptr buf, const cl_mem_flags& f = U_COPY_READ_WRITE) const {
			return std::make_shared<BufferCL<T>>(*_context, buf, f);
		}

		template <typename T>
		typename BufferCL<T>::Ptr NewReadOnlyBuffer(typename Storage<T>::Ptr buf) const {
			return std::make_shared<BufferCL<T>>(*_context, buf, U_COPY_READ);
		}

		template <typename T>
		typename BufferCL<T>::Ptr NewWriteOnlyBuffer(typename Storage<T>::Ptr buf) const {
			return std::make_shared<BufferCL<T>>(*_context, buf, U_WRITE);
		}


		ProgramCL::Ptr NewProgramFromFiles(const std::vector<std::string>& kernels);
		ProgramCL::Ptr NewProgramFromFile(const std::string& kernel);
		ProgramCL::Ptr NewProgramFromSource(const std::string& kernel);

	private:
        std::vector<DeviceCL::Ptr> _device;

		std::unique_ptr<cl::Context> _context;

		U_DISABLE_COPY_AND_ASSIGNMENT(ContextCL);
	};

	ProgramCL::Ptr ContextCL::NewProgramFromFiles(const std::vector<std::string>& kernels) {
		cl::Program::Sources source;
		for (const auto& str : kernels) {
			std::ifstream programFile(str);
			std::string programString(
				std::istreambuf_iterator<char>(programFile),
				(std::istreambuf_iterator<char>())
				);
			source.push_back(std::make_pair(programString.c_str(), programString.length()));
		}

		const cl::Context& c_ = Get();
		return std::make_shared<ProgramCL>(cl::Program(c_, source));
	}

	ProgramCL::Ptr ContextCL::NewProgramFromFile(const std::string& kernel) {
		std::ifstream programFile(kernel);
		std::string source(
			std::istreambuf_iterator<char>(programFile),
			(std::istreambuf_iterator<char>())
			);

		const cl::Context& c_ = Get();
		return std::make_shared<ProgramCL>(cl::Program(c_, source));
	}

	ProgramCL::Ptr ContextCL::NewProgramFromSource(const std::string& kernel) {
		const cl::Context& c_ = Get();
		return std::make_shared<ProgramCL>(cl::Program(c_, kernel, false));
	}

	std::vector<DeviceCL::Ptr> ContextCL::GPUs() {
		std::vector<DeviceCL::Ptr> t_;
        for (const auto& d : Devices()) {
			if (d->GetInfo().Type == CL_DEVICE_TYPE_GPU) {
				t_.push_back(d);
			}
		}
		return std::move(t_);
	}

	const DeviceCL& ContextCL::GPU() {
        for (const auto& d : Devices()) {
			if (d->GetInfo().Type == CL_DEVICE_TYPE_GPU) {
				return *d;
			}
		}
		throw std::runtime_error("0 gpus");
	}

	std::vector<DeviceCL::Ptr> ContextCL::CPUs() {
		std::vector<DeviceCL::Ptr> t_;
        for (const auto& d : Devices()) {
			if (d->GetInfo().Type == CL_DEVICE_TYPE_CPU) {
				t_.push_back(d);
			}
		}
		return std::move(t_);
	}

	const DeviceCL& ContextCL::CPU() {
        for (const auto& d : Devices()) {
			if (d->GetInfo().Type == CL_DEVICE_TYPE_CPU) {
				return *d;
			}
		}
		throw std::runtime_error("0 gpus");
	}

	class PlatformCL {
	public:
		typedef std::shared_ptr<PlatformCL> Ptr;
		typedef std::unique_ptr<PlatformCL> UPtr;

		struct Info {
			std::string Name;
		};

		PlatformCL(const cl::Platform& p = cl::Platform::getDefault()) {
			_platform = p;
			_info.Name = p.getInfo<CL_PLATFORM_NAME>();
		}

		const cl::Platform& Get() const { return _platform; }
		cl::Platform& Get() { return _platform; }

		const Info& GetInfo() const { return _info; }
		Info& GetInfo() { return _info; }

		ContextCL::Ptr NewDefaultContext() const {
			return ContextCL::Ptr(new ContextCL);
		}

		ContextCL::Ptr NewCPUContext() const {
			std::vector<cl::Device> d;
			_platform.getDevices(CL_DEVICE_TYPE_CPU, &d);

			return ContextCL::Ptr(new ContextCL(d));
		}

		ContextCL::Ptr NewGPUContext() const {
			std::vector<cl::Device> d;
			_platform.getDevices(CL_DEVICE_TYPE_GPU, &d);

			return ContextCL::Ptr(new ContextCL(d));
		}

		ContextCL::Ptr NewCompleteContext() const {
			std::vector<cl::Device> d;
			_platform.getDevices(CL_DEVICE_TYPE_ALL, &d);

			return ContextCL::Ptr(new ContextCL(d));
		}

		static ContextCL::Ptr NewContext() {
			return ContextCL::Ptr(new ContextCL);
		}

	private:
		cl::Platform _platform;
		Info _info;

		U_DISABLE_COPY_AND_ASSIGNMENT(PlatformCL);
	};

}}

#endif
