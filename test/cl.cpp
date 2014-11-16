#include <gtest/gtest.h>

#include <CL/ContextCL.h>
#include <CL/KernelCL.h>
#include <CL/QueueCL.h>
#include <CL/ProgramCL.h>

TEST(CL, ContextDefault) {
    try {
    GPU::CL::PlatformCL def;
    std::cout << def.GetInfo().Name << std::endl;

    auto context = def.NewDefaultContext();
    const auto& d = context->Device();
	std::cout << "Vendor: " << d.GetInfo().Vendor << std::endl;
	std::cout << "Max buffer size: " << d.GetInfo().MaxBufferSize << std::endl;
	ASSERT_EQ(d.GetInfo().Type, CL_DEVICE_TYPE_GPU);

    ASSERT_TRUE(d.IsDefault());
    } catch (const cl::Error& err) {
        TRACE(err.err(), err.what());
    }
}

TEST(CL, ContextMulti) {
	GPU::CL::ContextCL::Ptr cpuContext, gpuContext;

	{
		GPU::CL::PlatformCL platform;
		cpuContext = platform.NewCPUContext();
		gpuContext = platform.NewGPUContext();
	}

	const auto& d = cpuContext->Device();
	std::cout << " ** Vendor: " << d.GetInfo().Vendor << std::endl;
	std::cout << " ** Max buffer size: " << d.GetInfo().MaxBufferSize << std::endl;

	ASSERT_EQ(d.GetInfo().Type, CL_DEVICE_TYPE_CPU);

	const auto& g = gpuContext->Device();
	std::cout << " ** Vendor: " << g.GetInfo().Vendor << std::endl;
	std::cout << " ** Max buffer size: " << g.GetInfo().MaxBufferSize << std::endl;

	ASSERT_EQ(g.GetInfo().Type, CL_DEVICE_TYPE_GPU);
}

TEST(CL, ContextComplete) {
    GPU::CL::ContextCL::Ptr context = GPU::CL::PlatformCL().NewCompleteContext();

	auto gpus = context->GPUs();
	for (auto d : gpus) {
		std::cout << " ** Vendor: " << d->GetInfo().Vendor << std::endl;
		std::cout << " ** Max buffer size: " << d->GetInfo().MaxBufferSize << std::endl;
	}

	auto cpus = context->CPUs();
	for (auto d : cpus) {
		std::cout << " ** Vendor: " << d->GetInfo().Vendor << std::endl;
		std::cout << " ** Max buffer size: " << d->GetInfo().MaxBufferSize << std::endl;
	}
}

typedef struct {
    cl_float4 vec;
} UserObj;

TEST(CL, SimpleUserObject) {
	try {
		GPU::CL::ContextCL::Ptr gpuContext(new GPU::CL::ContextCL);

		GPU::CL::ProgramCL::Ptr program = gpuContext->NewProgramFromSource(
			U_KERNEL_CL(
				typedef struct {
					float4 vec;
				} UserObj;
				__kernel void empty(__global UserObj* obj) {}
			)
		);

		program->BuildFor(gpuContext->Device());

		auto kernel = program->NewKernel(gpuContext->Device(), "empty");

		UserObj obj[1024];

		auto storage = GPU::CL::RawPointer<UserObj>::New(obj, 1024);
		auto buf = gpuContext->NewBuffer<UserObj>(storage);
		kernel->Args(*buf);

		gpuContext.reset();
	} catch (const cl::Error&e) {
		TRACE(e.err(), e.what());
	}
}

TEST(CL, Buffer) {
	GPU::CL::ContextCL gpuContext;

	{
		float b[2] = { 2.0, 1.0 };
		GPU::CL::RawPointer<float> buf(b, 2);
		buf.At(0) = 1.0;
		ASSERT_DOUBLE_EQ(buf.At(0), 1.0);
		
		const float* cp = buf.Data();
		ASSERT_EQ(cp, b);

		GPU::CL::Storage<float>::Ptr base = GPU::CL::RawPointer<float>::New(b, 2);
		try {
			GPU::CL::BufferCL<float>::Ptr bufcl = gpuContext.NewBuffer<float>(base);
		}
		catch (const cl::Error& err) {
			TRACE(err.err(), err.what());
		}
	}

	{
		GPU::CL::Array<float, 10> buf;
		buf.At(0) = 1.0;
		ASSERT_DOUBLE_EQ(buf.At(0), 1.0);

		GPU::CL::Storage<float>::Ptr base = GPU::CL::Array<float, 10>::New();
		base->At(0) = 1.0;
		ASSERT_DOUBLE_EQ(base->At(0), 1.0);

		try {
			GPU::CL::BufferCL<float>::Ptr bufcl = gpuContext.NewBuffer<float>(base);
		}
		catch (const cl::Error& err) {
			TRACE(err.err(), err.what());
		}
	}

	{
		GPU::CL::Vector<float> buf(10);
		buf.PushBack(1.0);
		ASSERT_DOUBLE_EQ(buf.At(0), 1.0);
		
		GPU::CL::Storage<float>::Ptr base = GPU::CL::Vector<float>::New(10);
		base->PushBack(1.0);
		ASSERT_DOUBLE_EQ(base->At(0), 1.0);

		try {
			GPU::CL::BufferCL<float>::Ptr bufcl = gpuContext.NewBuffer<float>(base);
		}
		catch (const cl::Error& err) {
			TRACE(err.err(), err.what());
		}
	}

	{
		GPU::CL::Storage<float>::Ptr base = GPU::CL::ManagedBuffer<float>::New(new float[10], 10);
		ASSERT_EQ(base->Size(), 10);

		try {
			GPU::CL::BufferCL<float>::Ptr bufcl = gpuContext.NewBuffer<float>(GPU::CL::ManagedBuffer<float>::New(new float[10], 10));
		}
		catch (const cl::Error& err) {
			TRACE(err.err(), err.what());
		}
	}


    unsigned long long size = 4 * 1024 * 1024;
	std::vector<GPU::CL::BufferCL<cl_float>::Ptr> list;
	for (int i = 0; i < 2; i++) {
		try {
			GPU::CL::BufferCL<float>::Ptr bufcl = 
				gpuContext.NewBuffer<cl_float>(GPU::CL::ManagedBuffer<cl_float>::New(new cl_float[size], size));
			list.push_back(bufcl);
		}
		catch (const cl::Error& err) {
			TRACE(err.err(), err.what());
		}
	}
}

TEST(CL, Queue) {
	GPU::CL::ContextCL gpuContext;

	auto program = gpuContext.NewProgramFromSource(
		U_KERNEL_CL(
			__kernel void matrix(__global float4 *a, int i, float y) {
				__private size_t index = get_global_size(0) * get_global_id(1) + get_global_id(0);
				a[index] = index;
			}
		)
	);

	program->BuildFor(gpuContext.Device());

	try{
		auto kernel = program->NewKernel(gpuContext.Device(), "matrix");

		std::cout << " ** Allocating.. ";

        unsigned long long size = 1024;
		auto input = GPU::CL::ManagedBuffer<float>::New(new float[4 * size * size], 4 * size * size);
		GPU::CL::BufferCL<float>::Ptr buf = gpuContext.NewBuffer<float>(input);
		std::cout << 4 * size * size << " floats, " << input->RawSize() << " bytes." << std::endl;

		kernel->Args(*buf, 1, 1.0f);

		GPU::CL::QueueCL& cq = gpuContext.Device().Queue();

		GPU::CL::KernelCL::Range r;
		r.GlobalSize = cl::NDRange(size, size);
		r.LocalSize = cl::NDRange(8, 8);

		GPU::CL::EventCL ev([&size](cl_event, cl_int) {
			std::cout << "On event async! Clojure buffer width: "  << size << std::endl;
		});

		cq.FillBuffer(*buf, 1.0f);
		cq.ReadBuffer(*buf);
		ASSERT_FLOAT_EQ(input->At(0), 1.0);
		ASSERT_EQ(input->At(4 * size * size - 1), 1.0);

		cq.Enqueue(*kernel, r, ev);
		cq.ReadBuffer(*buf);

		ASSERT_EQ(input->At(0), 0.0);
		ASSERT_EQ(input->At(4), 1);
		ASSERT_EQ(input->At(4), input->At(7));
		ASSERT_EQ(input->At(4 * 2), 2);
		ASSERT_EQ(input->At(4 * 16), 16);

		ASSERT_EQ(input->At(4 * size * size - 1), size * size - 1);

		GPU::CL::BufferRectCL<float> rect(4, 4);
		rect.DeviceOrigin(4, 4);
		rect.HostOrigin(4, 0);
		rect.DevicePitch(size);

		cq.ReadBufferRect(*buf, rect);
		ASSERT_EQ(input->At(0), 0.0);
		ASSERT_EQ(input->At(3), 0.0);
        ASSERT_EQ(input->At(4), 4 * size + 1);

		rect.DeviceOrigin(1, 1);
		rect.HostOrigin(2, 0);
		cq.WriteBufferRect(*buf, rect);
        ASSERT_EQ(input->At(4), 4 * size + 1);
		cq.FillBuffer(*buf, 2.0f);
		cq.ReadBufferRect(*buf, rect);
		ASSERT_EQ(input->At(4), 2);

		cq.Finish();
	}
	catch (const cl::Error& err) {
		TRACE(err.err(), err.what());
	}
}
