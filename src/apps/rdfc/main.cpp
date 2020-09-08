//
// Created by Mike Smith on 2020/8/17.
//

#include <opencv2/opencv.hpp>

#include "feature_prefilter.h"
#include "gradient_magnitude.h"
#include "guided_nlm_filter.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

[[nodiscard]] cv::Mat load_image(const std::string &path) {
    LUISA_INFO("Loading image: \"", path, "\"...");
    auto image = cv::imread(path, cv::IMREAD_UNCHANGED);
    cv::cvtColor(image, image, image.channels() == 1 ? cv::COLOR_GRAY2RGBA : cv::COLOR_BGR2RGBA);
    const auto size = image.rows * image.cols * image.channels();
    for (auto i = 0; i < size; i++) {
        auto &&v = reinterpret_cast<float *>(image.data)[i];
        if (std::isnan(v)) { v = 0.0f; }
        if (std::isinf(v)) { v = 1e6f; }
    }
    return image;
}

void standardize(cv::Mat &feature_image, cv::Mat &feature_var_image, cv::Mat &feature_a_image, cv::Mat &feature_b_image) {
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(feature_image, mean, stddev);
    LUISA_INFO("Image mean = (", mean[0], ", ", mean[1], ", ", mean[2], "), stddev = (", stddev[0], ", ", stddev[1], ", ", stddev[2], ")");
    feature_image = (feature_image - mean) / stddev;
    feature_image = (feature_image - mean) / stddev;
    feature_a_image = (feature_a_image - mean) / stddev;
    feature_b_image = (feature_b_image - mean) / stddev;
    feature_var_image /= stddev * stddev;
}

int main(int argc, char *argv[]) {
    
    try {
        
        Context context{argc, argv};
        auto device = Device::create(&context);
        
        auto color_image = load_image(context.working_path("color.exr"));
        auto color_var_image = load_image(context.working_path("colorVariance.exr"));
    
        auto width = color_image.cols;
        auto height = color_image.rows;
        auto color = device->allocate_texture<float4>(width, height);
        auto color_var = device->allocate_texture<float4>(width, height);
        device->launch([&](Dispatcher &dispatch) {
            dispatch(color->copy_from(color_image.data));
            dispatch(color_var->copy_from(color_var_image.data));
        });
        
        auto feature = device->allocate_texture<float4>(width, height);
        auto feature_var = device->allocate_texture<float4>(width, height);
        auto feature_grad = device->allocate_texture<float4>(width, height);
        auto feature_a = device->allocate_texture<float4>(width, height);
        auto feature_b = device->allocate_texture<float4>(width, height);
        
        auto filter = std::make_unique<FeaturePrefilter>(
            *device, *feature, *feature_var, *feature_a, *feature_b,
            *feature, *feature_var, *feature_a, *feature_b);
        
        auto grad = std::make_unique<GradientMagnitude>(*device, *feature, *feature_grad);
        
        std::map<std::string, std::shared_ptr<Texture>> features;
        for (auto feature_name : {"albedo", "normal", "depth", "visibility"}) {
            
            auto feature_image = load_image(context.working_path(serialize(feature_name, ".exr")));
            auto feature_var_image = load_image(context.working_path(serialize(feature_name, "Variance.exr")));
            auto feature_a_image = load_image(context.working_path(serialize(feature_name, "A.exr")));
            auto feature_b_image = load_image(context.working_path(serialize(feature_name, "B.exr")));
            standardize(feature_image, feature_var_image, feature_a_image, feature_b_image);
            
            auto &&feature_out = *features.emplace(feature_name, device->allocate_texture<float4>(width, height)).first->second;
            auto &&feature_var_out = *features.emplace(serialize(feature_name, "_var"), device->allocate_texture<float4>(width, height)).first->second;
            auto &&feature_grad_out = *features.emplace(serialize(feature_name, "_grad"), device->allocate_texture<float4>(width, height)).first->second;
            
            device->launch([&](Dispatcher &dispatch) {
                dispatch(feature->copy_from(feature_image.data));
                dispatch(feature_var->copy_from(feature_var_image.data));
                dispatch(feature_a->copy_from(feature_a_image.data));
                dispatch(feature_b->copy_from(feature_b_image.data));
                dispatch(*filter);
                dispatch(*grad);
                dispatch(feature->copy_to(feature_out));
                dispatch(feature_var->copy_to(feature_var_out));
                dispatch(feature_grad->copy_to(feature_grad_out));
            }, [feature_name] {
                LUISA_INFO("Done filtering feature \"", feature_name, "\".");
            });
        }
        
        auto guided_nlm = std::make_unique<GuidedNonLocalMeansFilter>(
            *device, 5, 3, *color, *color_var, static_cast<float>(INFINITY), 1e-3f,
            *features["albedo"], *features["albedo_var"], *features["albedo_grad"], 0.6f,
            *features["normal"], *features["normal_var"], *features["normal_grad"], 0.6f,
            *features["depth"], *features["depth_var"], *features["depth_grad"], 0.6f,
            *features["visibility"], *features["visibility_var"], *features["visibility_grad"], 0.6f,
            *color);
        
        if (!std::filesystem::exists(context.working_path("rdfc"))) {
            std::filesystem::create_directories(context.working_path("rdfc"));
        }
        
        device->launch([&](Dispatcher &dispatch) {
            dispatch(*guided_nlm);
            dispatch(color->copy_to(color_image.data));
        }, [&] {
            cv::cvtColor(color_image, color_image, cv::COLOR_RGBA2BGR);
            cv::imwrite((context.working_path("rdfc") / "color.exr").string(), color_image);
            LUISA_INFO("Done saving filtered color.");
        });
        
        std::map<std::string, cv::Mat> feature_images;
        for (auto feature_name : {"albedo", "normal", "depth", "visibility"}) {
            auto &&feature_image = feature_images.emplace(feature_name, cv::Mat{}).first->second;
            auto &&feature_var_image = feature_images.emplace(serialize(feature_name, "_var"), cv::Mat{}).first->second;
            auto &&feature_grad_image = feature_images.emplace(serialize(feature_name, "_grad"), cv::Mat{}).first->second;
            feature_image.create(height, width, CV_32FC4);
            feature_var_image.create(height, width, CV_32FC4);
            feature_grad_image.create(height, width, CV_32FC4);
            device->launch([&](Dispatcher &dispatch) {
                dispatch(features[feature_name]->copy_to(feature_image.data));
                dispatch(features[serialize(feature_name, "_var")]->copy_to(feature_var_image.data));
                dispatch(features[serialize(feature_name, "_grad")]->copy_to(feature_grad_image.data));
            }, [feature_name, &context, &feature_images] {
                auto &&feature_image = feature_images[feature_name];
                auto &&feature_var_image = feature_images[serialize(feature_name, "_var")];
                auto &&feature_grad_image = feature_images[serialize(feature_name, "_grad")];
                cv::cvtColor(feature_image, feature_image, cv::COLOR_RGBA2BGR);
                cv::cvtColor(feature_var_image, feature_var_image, cv::COLOR_RGBA2BGR);
                cv::cvtColor(feature_grad_image, feature_grad_image, cv::COLOR_RGBA2BGR);
                cv::imwrite((context.working_path("rdfc") / serialize(feature_name, ".exr")).string(), feature_image);
                cv::imwrite((context.working_path("rdfc") / serialize(feature_name, "-variance.exr")).string(), feature_var_image);
                cv::imwrite((context.working_path("rdfc") / serialize(feature_name, "-gradient.exr")).string(), feature_grad_image);
            });
        }
        device->synchronize();
        
    } catch (const std::exception &e) {
        LUISA_ERROR("Caught exception: ", e.what());
    }
}
