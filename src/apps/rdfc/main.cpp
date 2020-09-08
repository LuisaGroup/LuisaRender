//
// Created by Mike Smith on 2020/8/17.
//

#include <opencv2/opencv.hpp>

#include "feature_prefilter.h"
#include "gradient_magnitude.h"

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
        
        std::unique_ptr<FeaturePrefilter> filter;
        std::shared_ptr<Texture> feature;
        std::shared_ptr<Texture> feature_var;
        std::shared_ptr<Texture> feature_a;
        std::shared_ptr<Texture> feature_b;
        std::map<std::string, std::shared_ptr<Texture>> features;
        
        auto width = 0;
        auto height = 0;
        
        for (auto feature_name : {"albedo", "normal", "depth", "visibility"}) {
            
            auto feature_image = load_image(context.working_path(serialize(feature_name, ".exr")));
            auto feature_var_image = load_image(context.working_path(serialize(feature_name, "Variance.exr")));
            auto feature_a_image = load_image(context.working_path(serialize(feature_name, "A.exr")));
            auto feature_b_image = load_image(context.working_path(serialize(feature_name, "B.exr")));
            standardize(feature_image, feature_var_image, feature_a_image, feature_b_image);
            
            if (filter == nullptr) {  // not initialized
                width = feature_image.cols;
                height = feature_image.rows;
                feature = device->allocate_texture<float4>(width, height);
                feature_var = device->allocate_texture<float4>(width, height);
                feature_a = device->allocate_texture<float4>(width, height);
                feature_b = device->allocate_texture<float4>(width, height);
                filter = std::make_unique<FeaturePrefilter>(
                    *device, *feature, *feature_var, *feature_a, *feature_b,
                    *feature, *feature_var, *feature_a, *feature_b);
            }
            
            auto &&feature_out = *features.emplace(feature_name, device->allocate_texture<float4>(width, height)).first->second;
            auto &&feature_var_out = *features.emplace(serialize(feature_name, "_var"), device->allocate_texture<float4>(width, height)).first->second;
            
            device->launch([&](Dispatcher &dispatch) {
                dispatch(feature->copy_from(feature_image.data));
                dispatch(feature_var->copy_from(feature_var_image.data));
                dispatch(feature_a->copy_from(feature_a_image.data));
                dispatch(feature_b->copy_from(feature_b_image.data));
                dispatch(*filter);
                dispatch(feature->copy_to(feature_out));
                dispatch(feature_var->copy_to(feature_var_out));
            }, [&] {
                LUISA_INFO("Done filtering feature \"", feature_name, "\".");
            });
        }
        
        std::map<std::string, cv::Mat> feature_images;
        for (auto feature_name : {"albedo", "normal", "depth", "visibility"}) {
            auto &&feature_image = feature_images.emplace(feature_name, cv::Mat{}).first->second;
            auto &&feature_var_image = feature_images.emplace(serialize(feature_name, "_var"), cv::Mat{}).first->second;
            feature_image.create(height, width, CV_32FC4);
            feature_var_image.create(height, width, CV_32FC4);
            device->launch([&](Dispatcher &dispatch) {
                dispatch(features[feature_name]->copy_to(feature_image.data));
                dispatch(features[serialize(feature_name, "_var")]->copy_to(feature_var_image.data));
            }, [feature_name, &context, &feature_images] {
                auto &&feature_image = feature_images[feature_name];
                auto &&feature_var_image = feature_images[serialize(feature_name, "_var")];
                cv::cvtColor(feature_image, feature_image, cv::COLOR_RGBA2BGR);
                cv::cvtColor(feature_var_image, feature_var_image, cv::COLOR_RGBA2BGR);
                cv::imwrite((context.working_path("rdfc") / serialize(feature_name, ".exr")).string(), feature_image);
                cv::imwrite((context.working_path("rdfc") / serialize(feature_name, "-variance.exr")).string(), feature_var_image);
            });
        }
        device->synchronize();
        
    } catch (const std::exception &e) {
        LUISA_ERROR("Caught exception: ", e.what());
    }
}
