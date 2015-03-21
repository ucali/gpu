#ifndef IMAGE_GPU_H
#define IMAGE_GPU_H

#include <memory>

namespace GPU {

	class Image {
	public:
		Image() {}
		virtual ~Image() {}

		virtual unsigned char* Bits() const = 0;
		virtual size_t Width() const = 0;
		virtual size_t Height() const = 0;
		virtual size_t BytesPerLine() const { return 0;  };
	};

}

#endif