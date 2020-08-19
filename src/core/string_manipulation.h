//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#include <string>
#include <string_view>
#include <sstream>
#include <fstream>
#include <array>
#include <filesystem>

#include "sha1.h"

namespace luisa { inline namespace utility {

template<typename ...Args>
inline std::string serialize(Args &&...args) noexcept {
    std::ostringstream ss;
    static_cast<void>((ss << ... << std::forward<Args>(args)));
    return ss.str();
}

inline auto sha1_digest(const std::string &s) noexcept {
    return SHA1{s}.digest();
}

inline std::string pascal_to_snake_case(std::string_view s) noexcept {  // TODO: Robustness
    std::string result;
    auto lower_met = false;
    for (auto i = 0u; i < s.size(); i++) {
        auto c = s[i];
        if (std::isupper(c)) {
            if (lower_met || (i != 0u && i != s.size() - 1u && !std::isupper(s[i + 1]))) { result.push_back('_'); }
            lower_met = false;
        } else {
            lower_met = true;
        }
        result.push_back(std::tolower(c));
    }
    return result;
}

inline std::string text_file_contents(const std::filesystem::path &file_path) {
    std::ifstream file{file_path};
    if (!file.is_open()) {
        std::ostringstream ss;
        ss << "Failed to open file: " << file_path;
        throw std::runtime_error{ss.str()};
    }
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

}}

#ifdef __OBJC__

#import <Foundation/Foundation.h>

namespace luisa { inline namespace utility {

[[nodiscard]] inline NSString *make_objc_string(const char *s) noexcept {
    return [[NSString alloc] initWithCString:s encoding:NSUTF8StringEncoding];
}

[[nodiscard]] inline NSString *make_objc_string(std::string_view sv) noexcept {
    return [[NSString alloc] initWithBytes:sv.data() length:sv.size() encoding:NSUTF8StringEncoding];
}

[[nodiscard]] inline std::string to_string(NSString *s) noexcept {
    return s.UTF8String;
}

[[nodiscard]] inline std::string_view to_string_view(NSString *s) noexcept {
    return {s.UTF8String, s.length};
}

}}

#endif
