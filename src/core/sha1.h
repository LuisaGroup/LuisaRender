// From: https://github.com/vog/sha1

/*
    sha1.hpp - source code of

    ============
    SHA-1 in C++
    ============

    100% Public Domain.

    Original C Code
        -- Steve Reid <steve@edmweb.com>
    Small changes to fit into bglibs
        -- Bruce Guenter <bruce@untroubled.org>
    Translation to simpler C++ Code
        -- Volker Diels-Grabsch <v@njh.eu>
    Safety fixes
        -- Eugene Hopkinson <slowriot at voxelstorm dot com>
    Header-only library
        -- Zlatko Michailov <zlatko@michailov.org>
*/

#pragma once

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <array>

class SHA1 {

public:
    using Digest = std::array<uint32_t, 5>;

private:
    Digest _digest;
    std::string _buffer;
    uint64_t _transforms;
    
public:
    explicit SHA1(const std::string &s);
    [[nodiscard]] auto digest() const noexcept { return _digest; }
};
