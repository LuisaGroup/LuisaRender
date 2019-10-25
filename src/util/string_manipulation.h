//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#ifdef __OBJC__

#import <string>
#import <string_view>
#import <Foundation/Foundation.h>

[[nodiscard]] inline NSString *make_objc_string(const char *s) noexcept {
    return [[[NSString alloc] initWithCString:s encoding:NSUTF8StringEncoding] autorelease];
}

[[nodiscard]] inline NSString *make_objc_string(std::string_view sv) noexcept {
    return [[[NSString alloc] initWithBytes:sv.data() length:sv.size() encoding:NSUTF8StringEncoding] autorelease];
}

[[nodiscard]] inline std::string to_string(NSString *s) noexcept {
    return s.UTF8String;
}

[[nodiscard]] inline std::string_view to_string_view(NSString *s) noexcept {
    return {s.UTF8String, s.length};
}

#endif
