//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#ifdef __OBJC__

#include <string_view>
#include <Foundation/Foundation.h>

inline NSString *make_objc_string(const char *s) noexcept {
    return [[[NSString alloc] initWithCString:s encoding:NSUTF8StringEncoding] autorelease];
}

#endif
