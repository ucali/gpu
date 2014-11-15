#ifndef QUEUE_CL_H
#define QUEUE_CL_H

#include "KernelCL.h"

#include <functional>

namespace GPU {
	namespace CL {

	class EventCL {
	public:
		typedef void (CL_CALLBACK* _FUNCTOR_CALLBACK) (cl_event, cl_int, void*);
		typedef std::function<void(cl_event, cl_int)> Func;

		EventCL(const Func& f) : _ptr(this) {
			Callback = f;
			_c = [](cl_event ev, cl_int i, void* p) {
				static_cast<EventCL*>(p)->Callback(ev, i);
			};
		}

		EventCL(const _FUNCTOR_CALLBACK& c, void* ptr) : _ptr(ptr) {
			_c = c;
		}

		cl::Event* Event() { return &_ev; }

		std::function<void (cl_event, cl_int)> Callback;

	private:
		void* _ptr;
		cl::Event _ev;
		_FUNCTOR_CALLBACK _c;

		void _Set() { _ev.setCallback(CL_COMPLETE, _c, _ptr); }

		friend class QueueCL; 

		U_DISABLE_COPY_AND_ASSIGNMENT(EventCL);
	};

	class QueueCL {
	public:
		typedef std::shared_ptr<QueueCL> Ptr;

		void Enqueue(const KernelCL& k) { _queue.enqueueTask(k.Get()); }
		void Enqueue(const KernelCL& k, EventCL& ev) { _queue.enqueueTask(k.Get(), nullptr, ev.Event()); ev._Set();  }

		void Enqueue(const KernelCL& k, const KernelCL::Range& r) {
			_queue.enqueueNDRangeKernel(k.Get(), r.Offset, r.GlobalSize, r.LocalSize);
		}

		void Enqueue(const KernelCL& k, const KernelCL::Range& r, EventCL& ev) {
			_queue.enqueueNDRangeKernel(k.Get(), r.Offset, r.GlobalSize, r.LocalSize, nullptr, ev.Event());
			ev._Set();
		}

		/**** Generic buffer operations ****/

		template <typename T>
		void FillBuffer(const BufferCL<T>& src, const T& val) {
			_queue.enqueueFillBuffer<T>(src.Get(), val, src.DeviceBytesOffset(), src.DeviceSizeFromOffset());
		}

		void Flush() { _queue.flush(); }
		void Finish() { _queue.finish(); }
		
		/**** Buffer write operations ****/

		template <typename T>
		void WriteBuffer(const BufferCL<T>& src) {
			_queue.enqueueWriteBuffer(src.Get(), CL_TRUE, src.HostBytesOffset(), src.HostSizeFromOffset(), src.Data());
		}

		template <typename T>
		void WriteBuffer(const BufferCL<T>& src, EventCL& ev) {
			_queue.enqueueWriteBuffer(src.Get(), CL_FALSE, src.HostBytesOffset(), src.HostSizeFromOffset(), src.Data(), ev.Event());
			ev._Set();
		}

		template <typename T>
		void WriteBufferRect(const BufferCL<T>& src, const BufferRectCL<T>& b) {
			_queue.enqueueWriteBufferRect(
				src.Get(), CL_TRUE,
				b.DeviceOrigin(), b.HostOrigin(),
				b.Region,
				b.DeviceRow, b.DeviceSlice,
				b.HostRow, b.HostSlice,
				src.Data()
			);
		}

		/**** Buffer read operations ****/

		template <typename T>
		void ReadBuffer(const BufferCL<T>& dst) {
			_queue.enqueueReadBuffer(dst.Get(), CL_TRUE, dst.DeviceBytesOffset(), dst.DeviceSizeFromOffset(), dst.Data());
		}

		template <typename T>
		void ReadBuffer(const BufferCL<T>& src, EventCL& ev) {
			_queue.enqueueReadBuffer(src.Get(), CL_FALSE, src.DeviceBytesOffset(), src.DeviceSizeFromOffset(), src.Data(), ev.Event());
			ev._Set();
		}

		template <typename T>
		void ReadBufferRect(const BufferCL<T>& src, const BufferRectCL<T>& b) {
			_queue.enqueueReadBufferRect(
				src.Get(), CL_TRUE,
				b.DeviceOrigin(), b.HostOrigin(),
				b.Region,
				b.DeviceRow, b.DeviceSlice,
				b.HostRow, b.HostSlice,
				src.Data()
			);
		}

		/*Buffer copy operations */

		template <typename T> 
		void CopyBuffer(const BufferCL<T>& src, BufferCL<T>& dest, const size_t& s = 0) {
			_queue.enqueueCopyBuffer(
				src.Get(), 
				dest.Get(), 
				src.DeviceBytesOffset(), 
				dest.DeviceBytesOffset(), 
				s ? s : src.DeviceSizeFromOffset()
			);
		}

		template <typename T>
		void CopyBufferRect(const BufferCL<T>& src, BufferCL<T>& dest, const BufferRectCL<T>& b) {
			_queue.enqueueCopyBufferRect(
				src.Get(),
				dest.Get(),
				b.DeviceOrigin(), b.HostOrigin(),
				b.Region,
				b.DeviceRow, b.DeviceSlice,
				b.HostRow, b.HostSlice
			);
		}

	private: 
		QueueCL(const cl::Context& c, const cl::Device& dev) {
			_queue = cl::CommandQueue(c, dev);
		}

		cl::CommandQueue _queue;

		friend class DeviceCL;
		U_DISABLE_COPY_AND_ASSIGNMENT(QueueCL);
	};
	
	}
}
#endif