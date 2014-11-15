#ifndef PROGRAM_CL_H 
#define PROGRAM_CL_H

#include "CommonCL.h"
#include "KernelCL.h"
#include "DeviceCL.h"

#include <fstream>

namespace GPU {
namespace CL {

	class ProgramCL {
	public:
		typedef std::shared_ptr<ProgramCL> Ptr;

		void BuildFor(const DeviceCL&);
		void BuildFor(const std::vector<DeviceCL::Ptr>&);

		KernelCL::Ptr NewKernel(const DeviceCL& c, const std::string& kernel);
		KernelCL::Ptr NewKernel(const std::string& kernel);

		const cl::Program& Get() const { return _program; }

		ProgramCL(const cl::Program& p) : _program(p) {}
	private:

		cl::Program _program;

		U_DISABLE_COPY_AND_ASSIGNMENT(ProgramCL);
	};

	void ProgramCL::BuildFor(const DeviceCL& d) {
		const cl::Device& d_ = d.Get();

		std::vector<cl::Device> list; list.push_back(d_);
		_program.build(list);
	}

	void ProgramCL::BuildFor(const std::vector<DeviceCL::Ptr>& dev) {
		std::vector<cl::Device> list; 
		for (auto d : dev) {
			list.push_back(d->Get());
		}
		_program.build(list);
	}

	KernelCL::Ptr ProgramCL::NewKernel(const DeviceCL& d, const std::string& kernel) {
		return std::make_shared<KernelCL>(d.Get(), Get(), kernel);
	}

	KernelCL::Ptr ProgramCL::NewKernel(const std::string& kernel) {
		return std::make_shared<KernelCL>(Get(), kernel);
	}

}}


#endif