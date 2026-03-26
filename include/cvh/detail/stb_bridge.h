#ifndef CVH_DETAIL_STB_BRIDGE_H
#define CVH_DETAIL_STB_BRIDGE_H

// Keep stb symbols TU-local so this header can be safely included by multiple
// translation units in header-only mode.
#ifndef STB_IMAGE_STATIC
#define STB_IMAGE_STATIC
#endif
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "../3rdparty/std/stb_image.h"

#ifndef STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_STATIC
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "../3rdparty/std/stb_image_write.h"

#endif  // CVH_DETAIL_STB_BRIDGE_H
