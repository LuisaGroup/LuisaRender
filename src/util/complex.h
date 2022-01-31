//
// Created by Mike Smith on 2022/1/31.
//

#pragma once

namespace luisa::render {

template<typename T>
struct Complex {
    T re, im;
    Complex(T re) : re{re}, im{0.0f} {}
    Complex(T re, T im) : re{re}, im{im} {}
    [[nodiscard]] Complex operator-() const noexcept { return {-re, -im}; }
    [[nodiscard]] Complex operator+(Complex z) const noexcept { return {re + z.re, im + z.im}; }
    [[nodiscard]] Complex operator-(Complex z) const noexcept { return {re - z.re, im - z.im}; }
    [[nodiscard]] Complex operator*(Complex z) const noexcept { return {re * z.re - im * z.im, re * z.im + im * z.re}; }
    [[nodiscard]] Complex operator/(Complex z) const { return Complex{(re * z.re + im * z.im), (im * z.re - re * z.im)} * (1.0f / (z.re * z.re + z.im * z.im)); }
    [[nodiscard]] friend Complex operator+(T value, Complex z) noexcept { return Complex(value) + z; }
    [[nodiscard]] friend Complex operator-(T value, Complex z) noexcept { return Complex(value) - z; }
    [[nodiscard]] friend Complex operator*(T value, Complex z) noexcept { return Complex(value) * z; }
    [[nodiscard]] friend Complex operator/(T value, Complex z) noexcept { return Complex(value) / z; }
};

}// namespace luisa::render
