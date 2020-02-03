//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#include <string>
#include <string_view>
#include <sstream>

namespace luisa { inline namespace utility {

template<typename ...Args>
inline std::string serialize(Args &&...args) noexcept {
    std::ostringstream ss;
    static_cast<void>((ss << ... << std::forward<Args>(args)));
    return ss.str();
}

}}

#ifdef __OBJC__

namespace luisa { inline namespace utility {

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

}}

#endif
