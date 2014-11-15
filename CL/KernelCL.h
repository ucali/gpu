#ifndef KERNEL_CL_H 
#define KERNEL_CL_H

#include "BufferCL.h"

#include <fstream>

namespace GPU {
namespace CL {

	class KernelCL {
	public:
		typedef std::shared_ptr<KernelCL> Ptr;

		struct Info {
			size_t maxWorkGroupSize;
			size_t preferredWorkGroupSizeMultiple;
			cl_ulong localMemSize;
			cl_ulong privateMemSize;
		};

		struct Range {
			Range() : Offset(cl::NullRange), GlobalSize(cl::NullRange), LocalSize(cl::NullRange) { }

			cl::NDRange Offset;
			cl::NDRange GlobalSize;
			cl::NDRange LocalSize;
		};

		KernelCL(const cl::Program& p, const std::string& n) : _name(n) {
			_kernel = cl::Kernel(p, _name.c_str());
		}

		KernelCL(const cl::Device& dev, const cl::Program& p, const std::string& n) : KernelCL(p, n) {
			_info.maxWorkGroupSize = _kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(dev);
			_info.preferredWorkGroupSizeMultiple = _kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(dev);
			_info.localMemSize = _kernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(dev);
			_info.privateMemSize = _kernel.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(dev);
		}

		const cl::Kernel& Get() const { return _kernel; }

		template <typename T, typename... P>
		void Args(const T& t, const P& ... args) { _Args(0, t, args...); }
		
		template <typename T>
		void Arg(cl_uint i, const T& t) {
			_kernel.setArg(i, t);
		}

		template <typename T>
		void Arg(cl_uint i, const BufferCL<T>& b) {
			_kernel.setArg(i, b.Get());
		}

	private:
		template <typename T, typename... P>
		void _Args(cl_uint i, const T& t, const P& ... args) {
			Arg(i, t);
			if (sizeof...(args)) {
				_Args(i + 1, args...);
			}
		}

		template <class T>
		void _Args(cl_uint i, const T& t) {
			Arg(i, t);
		}

		cl::Kernel _kernel;
		const std::string _name;
		Info _info;

		U_DISABLE_COPY_AND_ASSIGNMENT(KernelCL);
	};

}}


#endif