//
// Created by Mike Smith on 2020/8/19.
//

#include <array>
#include <cstring>
#include <iomanip>

// From: https://stackoverflow.com/questions/58524805/is-there-a-way-to-create-a-stringstream-from-a-string-view-without-copying-data
template<typename CharT, class TraitsT>
class view_streambuf final : public std::basic_streambuf<CharT, TraitsT> {
private:
    typedef std::basic_streambuf<CharT, TraitsT> super_type;
    typedef view_streambuf<CharT, TraitsT> self_type;
public:
    
    /**
    *  These are standard types.  They permit a standardized way of
    *  referring to names of (or names dependent on) the template
    *  parameters, which are specific to the implementation.
    */
    typedef typename super_type::char_type char_type;
    typedef typename super_type::traits_type traits_type;
    typedef typename traits_type::int_type int_type;
    typedef typename traits_type::pos_type pos_type;
    typedef typename traits_type::off_type off_type;
    
    typedef typename std::basic_string_view<char_type, traits_type> source_view;
    
    explicit view_streambuf(const source_view &src) noexcept:
        super_type(),
        src_(src) {
        auto buff = const_cast<char_type *>( src_.data());
        this->setg(buff, buff, buff + src_.length());
    }
    
    std::streamsize xsgetn(char_type *s, std::streamsize n) override {
        if (0 == n) {
            return 0;
        }
        if ((this->gptr() + n) >= this->egptr()) {
            n = this->egptr() - this->gptr();
            if (0 == n && !traits_type::not_eof(this->underflow())) {
                return -1;
            }
        }
        std::memmove(static_cast<void *>(s), this->gptr(), n);
        this->gbump(static_cast<int>(n));
        return n;
    }
    
    int_type pbackfail(int_type c) override {
        char_type *pos = this->gptr() - 1;
        *pos = traits_type::to_char_type(c);
        this->pbump(-1);
        return 1;
    }
    
    int_type underflow() override {
        return traits_type::eof();
    }
    
    std::streamsize showmanyc() override {
        return static_cast<std::streamsize>( this->egptr() - this->gptr());
    }
    
    ~view_streambuf() override = default;
    
private:
    const source_view &src_;
};

template<typename _char_type>
class view_istream final : public std::basic_istream<_char_type, std::char_traits<_char_type>> {
public:
    view_istream(const view_istream &) = delete;
    view_istream &operator=(const view_istream &) = delete;
private:
    typedef std::basic_istream<_char_type, std::char_traits<_char_type>> super_type;
    typedef view_streambuf<_char_type, std::char_traits<_char_type>> streambuf_type;
public:
    typedef _char_type char_type;
    typedef typename super_type::int_type int_type;
    typedef typename super_type::pos_type pos_type;
    typedef typename super_type::off_type off_type;
    typedef typename super_type::traits_type traits_type;
    typedef typename streambuf_type::source_view source_view;
    
    explicit view_istream(const source_view &src) :
        super_type(nullptr),
        sb_(nullptr) {
        sb_ = new streambuf_type(src);
        this->init(sb_);
    }
    
    view_istream(view_istream &&other) noexcept:
        super_type(std::forward<view_istream>(other)),
        sb_(std::move(other.sb_)) {}
    
    view_istream &operator=(view_istream &&rhs) noexcept {
        view_istream(std::forward<view_istream>(rhs)).swap(*this);
        return *this;
    }
    
    ~view_istream() override {
        delete sb_;
    }

private:
    streambuf_type *sb_;
};

// From: https://github.com/vog/sha1
#include "sha1.h"

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

constexpr size_t block_ints = 16;  /* number of 32bit integers per SHA1 block */
constexpr size_t block_bytes = block_ints * 4;

void reset(uint32_t digest[], std::string &buffer, uint64_t &transforms) {
    /* SHA1 initialization constants */
    digest[0] = 0x67452301;
    digest[1] = 0xefcdab89;
    digest[2] = 0x98badcfe;
    digest[3] = 0x10325476;
    digest[4] = 0xc3d2e1f0;
    
    /* Reset counters */
    buffer = "";
    transforms = 0;
}

uint32_t rol(const uint32_t value, const size_t bits) {
    return (value << bits) | (value >> (32u - bits));
}

uint32_t blk(const uint32_t block[block_ints], const size_t i) {
    return rol(block[(i + 13u) & 15u] ^ block[(i + 8u) & 15u] ^ block[(i + 2u) & 15u] ^ block[i], 1u);
}

/*
 * (R0+R1), R2, R3, R4 are the different operations used in SHA1
 */

void R0(const uint32_t block[block_ints], const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i) {
    z += ((w & (x ^ y)) ^ y) + block[i] + 0x5a827999 + rol(v, 5);
    w = rol(w, 30);
}

void R1(uint32_t block[block_ints], const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i) {
    block[i] = blk(block, i);
    z += ((w & (x ^ y)) ^ y) + block[i] + 0x5a827999 + rol(v, 5);
    w = rol(w, 30);
}

void R2(uint32_t block[block_ints], const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i) {
    block[i] = blk(block, i);
    z += (w ^ x ^ y) + block[i] + 0x6ed9eba1 + rol(v, 5);
    w = rol(w, 30);
}

void R3(uint32_t block[block_ints], const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i) {
    block[i] = blk(block, i);
    z += (((w | x) & y) | (w & x)) + block[i] + 0x8f1bbcdc + rol(v, 5);
    w = rol(w, 30);
}

void R4(uint32_t block[block_ints], const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i) {
    block[i] = blk(block, i);
    z += (w ^ x ^ y) + block[i] + 0xca62c1d6 + rol(v, 5);
    w = rol(w, 30);
}

/*
 * Hash a single 512-bit block. This is the core of the algorithm.
 */

void transform(uint32_t digest[], uint32_t block[block_ints], uint64_t &transforms) {
    /* Copy digest[] to working vars */
    uint32_t a = digest[0];
    uint32_t b = digest[1];
    uint32_t c = digest[2];
    uint32_t d = digest[3];
    uint32_t e = digest[4];
    
    /* 4 rounds of 20 operations each. Loop unrolled. */
    R0(block, a, b, c, d, e, 0);
    R0(block, e, a, b, c, d, 1);
    R0(block, d, e, a, b, c, 2);
    R0(block, c, d, e, a, b, 3);
    R0(block, b, c, d, e, a, 4);
    R0(block, a, b, c, d, e, 5);
    R0(block, e, a, b, c, d, 6);
    R0(block, d, e, a, b, c, 7);
    R0(block, c, d, e, a, b, 8);
    R0(block, b, c, d, e, a, 9);
    R0(block, a, b, c, d, e, 10);
    R0(block, e, a, b, c, d, 11);
    R0(block, d, e, a, b, c, 12);
    R0(block, c, d, e, a, b, 13);
    R0(block, b, c, d, e, a, 14);
    R0(block, a, b, c, d, e, 15);
    R1(block, e, a, b, c, d, 0);
    R1(block, d, e, a, b, c, 1);
    R1(block, c, d, e, a, b, 2);
    R1(block, b, c, d, e, a, 3);
    R2(block, a, b, c, d, e, 4);
    R2(block, e, a, b, c, d, 5);
    R2(block, d, e, a, b, c, 6);
    R2(block, c, d, e, a, b, 7);
    R2(block, b, c, d, e, a, 8);
    R2(block, a, b, c, d, e, 9);
    R2(block, e, a, b, c, d, 10);
    R2(block, d, e, a, b, c, 11);
    R2(block, c, d, e, a, b, 12);
    R2(block, b, c, d, e, a, 13);
    R2(block, a, b, c, d, e, 14);
    R2(block, e, a, b, c, d, 15);
    R2(block, d, e, a, b, c, 0);
    R2(block, c, d, e, a, b, 1);
    R2(block, b, c, d, e, a, 2);
    R2(block, a, b, c, d, e, 3);
    R2(block, e, a, b, c, d, 4);
    R2(block, d, e, a, b, c, 5);
    R2(block, c, d, e, a, b, 6);
    R2(block, b, c, d, e, a, 7);
    R3(block, a, b, c, d, e, 8);
    R3(block, e, a, b, c, d, 9);
    R3(block, d, e, a, b, c, 10);
    R3(block, c, d, e, a, b, 11);
    R3(block, b, c, d, e, a, 12);
    R3(block, a, b, c, d, e, 13);
    R3(block, e, a, b, c, d, 14);
    R3(block, d, e, a, b, c, 15);
    R3(block, c, d, e, a, b, 0);
    R3(block, b, c, d, e, a, 1);
    R3(block, a, b, c, d, e, 2);
    R3(block, e, a, b, c, d, 3);
    R3(block, d, e, a, b, c, 4);
    R3(block, c, d, e, a, b, 5);
    R3(block, b, c, d, e, a, 6);
    R3(block, a, b, c, d, e, 7);
    R3(block, e, a, b, c, d, 8);
    R3(block, d, e, a, b, c, 9);
    R3(block, c, d, e, a, b, 10);
    R3(block, b, c, d, e, a, 11);
    R4(block, a, b, c, d, e, 12);
    R4(block, e, a, b, c, d, 13);
    R4(block, d, e, a, b, c, 14);
    R4(block, c, d, e, a, b, 15);
    R4(block, b, c, d, e, a, 0);
    R4(block, a, b, c, d, e, 1);
    R4(block, e, a, b, c, d, 2);
    R4(block, d, e, a, b, c, 3);
    R4(block, c, d, e, a, b, 4);
    R4(block, b, c, d, e, a, 5);
    R4(block, a, b, c, d, e, 6);
    R4(block, e, a, b, c, d, 7);
    R4(block, d, e, a, b, c, 8);
    R4(block, c, d, e, a, b, 9);
    R4(block, b, c, d, e, a, 10);
    R4(block, a, b, c, d, e, 11);
    R4(block, e, a, b, c, d, 12);
    R4(block, d, e, a, b, c, 13);
    R4(block, c, d, e, a, b, 14);
    R4(block, b, c, d, e, a, 15);
    
    /* Add the working vars back into digest[] */
    digest[0] += a;
    digest[1] += b;
    digest[2] += c;
    digest[3] += d;
    digest[4] += e;
    
    /* Count the number of transformations */
    transforms++;
}

void buffer_to_block(const std::string &buffer, uint32_t block[block_ints]) {
    /* Convert the std::string (byte buffer) to a uint32_t array (MSB) */
    for (size_t i = 0; i < block_ints; i++) {
        block[i] = (buffer[4 * i + 3] & 0xff)
                   | (buffer[4 * i + 2] & 0xff) << 8
                   | (buffer[4 * i + 1] & 0xff) << 16
                   | (buffer[4 * i + 0] & 0xff) << 24;
    }
}

SHA1::SHA1(std::string_view s) {
    reset(_digest.data(), _buffer, _transforms);
    view_istream<char> is{s};
    while (true) {
        char sbuf[block_bytes];
        is.read(sbuf, block_bytes - _buffer.size());
        _buffer.append(sbuf, (std::size_t)is.gcount());
        if (_buffer.size() != block_bytes) { return; }
        uint32_t block[block_ints];
        buffer_to_block(_buffer, block);
        transform(_digest.data(), block, _transforms);
        _buffer.clear();
    }
}
