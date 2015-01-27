#include <CL\Image.h>
#include <FreeImage.h>

class FreeImage : public GPU::Image {
public:
	FreeImage() {}
	virtual ~FreeImage() {}

	virtual unsigned char* Bits() const { return NULL;  };
	virtual size_t Width() const { return 0;  };
	virtual size_t Height() const { return 0;  };
	virtual size_t BytesPerLine() const { return 0; };

private:

};