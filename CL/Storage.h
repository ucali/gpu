#ifndef STORAGE_GPU_H
#define STORAGE_GPU_H

#include <array>
#include <memory>

namespace GPU {
	namespace CL {

	template <typename T>
	class Storage {
	public:
		typedef std::shared_ptr<Storage<T>> Ptr;

		Storage() : _size(0) {}
		Storage(size_t s) : _size(s) {}

		virtual ~Storage() {}

		virtual T* Data() = 0;
		virtual const T* Data() const = 0;
		virtual void Release() = 0;

		virtual T& At(size_t) = 0;
		virtual const T& At(size_t) const = 0;

		virtual size_t Size() const { return _size; }

		virtual void PushBack(const T&) { throw std::runtime_error("PushBack, invalid operation."); }

		size_t RawSize() const { return _size * sizeof(T); }

	protected:
		void size(size_t s) { _size = s; }
	
	private:
		size_t _size;

		U_DISABLE_COPY_AND_ASSIGNMENT(Storage);
	};

	template <typename T>
	class ManagedBuffer : public Storage<T> {
	public:
		ManagedBuffer(T* t, size_t s) : Storage<T>(s), _pointer(t) {}

		virtual T* Data() {
			return _pointer.get();
		}

		virtual const T* Data() const {
			return _pointer.get();
		}

		virtual void Release() {
			_pointer.reset();
            this->size(0);
		}

		virtual T& At(size_t i) { return _pointer.get()[i]; }
		virtual const T& At(size_t i) const { return _pointer.get()[i]; }

        static typename Storage<T>::Ptr New(T* t, size_t s) {
			return std::make_shared<ManagedBuffer>(t, s);
		}

	private:
		std::shared_ptr<T> _pointer;
	};

	template <typename T>
	class RawPointer : public Storage<T> {
	public:
		RawPointer(T* t, size_t s) : Storage<T>(s), _pointer(t) {}

		virtual T* Data() {
			return _pointer;
		}

		virtual const T* Data() const {
			return _pointer;
		}

		virtual void Release() {} //TODO: (think) should this throw?

		virtual T& At(size_t i) { return _pointer[i]; }
		virtual const T& At(size_t i) const { return _pointer[i]; }

        static typename Storage<T>::Ptr New(T* t, size_t s) {
			return std::make_shared<RawPointer>(t, s);
		}

	private:
		T* _pointer;
	};

	template <typename T, size_t s>
	class Array : public Storage<T> {
	public:
		Array() : Storage<T>(s) {
            _pointer.reset(new std::array<T, s>());
		}

		virtual T* Data() {
			return &(_pointer->data()[0]);
		}

		virtual const T* Data() const {
			return _pointer->data();
		}

		virtual void Release() {
			_pointer.reset();
            this->size(0);
		} 

        static typename Storage<T>::Ptr New() {
			return std::make_shared<Array>();
		}

		virtual T& At(size_t i) { return _pointer->at(i); }
		virtual const T& At(size_t i) const { return _pointer->at(i); }

	private:
		std::unique_ptr<std::array<T, s>> _pointer;
	};

	template <typename T>
	class Vector : public Storage<T> {
	public:
		Vector() : Storage<T>(0) {}
		Vector(size_t s) : Storage<T>(s) {
			_vec.reserve(s);
		}

		virtual void PushBack(const T& t) { _vec.push_back(t); }

		virtual T* Data() {
			return &_vec[0];
		}

		virtual const T* Data() const {
			return _vec.data();
		}

		virtual void Release() {
			_vec.clear();
			_vec.swap(std::vector<T>());
		}

        static typename Storage<T>::Ptr New(size_t s) {
			return std::make_shared<Vector>(s);
		}

		virtual T& At(size_t i) { return _vec.at(i); }
		virtual const T& At(size_t i) const { return _vec.at(i); }

		virtual size_t Size() const { return _vec.size(); }

	private:
		std::vector<T> _vec;
	};

}}
#endif
