#include <cfloat>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/psda_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class PSDALayerTest : public CPUDeviceTest<Dtype> {
 protected:
  PSDALayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        num_labels_(3) {
    vector<int> shape(2);
    num_samples_ = num_labels_ * 500;
    shape[0] = num_samples_;
    shape[1] = num_labels_;
    blob_bottom_data_->Reshape(shape);
    shape.resize(1);
    blob_bottom_label_->Reshape(shape);
    FillBottoms();

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual void FillBottoms() {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
    caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(rng->generator());
    Dtype* bottom_data = blob_bottom_data_->mutable_cpu_data();
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      label_data[i] = (*prefetch_rng)() % num_labels_;
      std::vector<Dtype> rand_value(num_labels_);
      Dtype offset = Dtype(1.0);
      caffe_rng_gaussian<Dtype>(num_labels_, label_data[i] + offset,
                                label_data[i] + offset, &rand_value[0]);
      for (int k = 0; k < num_labels_; ++k) {
        bottom_data[i * num_labels_ + k] = static_cast<Dtype>(rand_value[k]);
      }
    }
  }

  void ComputeGroundTruth() {
    // Let's compute the ground truth. First we need some containers for results
    objs_mean_.resize(this->num_labels_);
    objs_std_dev_.resize(this->num_labels_);
    not_objs_mean_.resize(this->num_labels_);
    not_objs_std_dev_.resize(this->num_labels_);
    objs_count_.resize(this->num_labels_);
    not_objs_count_.resize(this->num_labels_);
    // Use direct method to compute mean and variance
    for (int i = 0; i < this->blob_bottom_label_->count(); ++i) {
      int label_value = this->blob_bottom_label_->data_at(i, 0, 0, 0);
      for (int k = 0; k < this->blob_bottom_data_->shape(1); ++k) {
        if (k == label_value) {
          objs_mean_[k] += this->blob_bottom_data_->data_at(i, k, 0, 0);
          objs_count_[k] += 1;
        } else {
          not_objs_mean_[k] += this->blob_bottom_data_->data_at(i, k, 0, 0);
          not_objs_count_[k] += 1;
        }
      }
    }

    for (int i = 0; i < objs_mean_.size(); ++i) {
      objs_mean_[i] = objs_mean_[i] / static_cast<Dtype>(objs_count_[i]);
      not_objs_mean_[i] =
          not_objs_mean_[i] / static_cast<Dtype>(not_objs_count_[i]);
      //     LOG(INFO) << "Count for class " << i << " is: " << objs_count[i];
      //           LOG(INFO) << "Mean of positive examples for class " << i
      //                     << " is: " << objs_mean_[i];
      //           LOG(INFO) << "Mean of negative examples for class " << i
      //                     << " is: " << not_objs_mean_[i];
    }

    for (int i = 0; i < this->blob_bottom_label_->count(); ++i) {
      int label_value = this->blob_bottom_label_->data_at(i, 0, 0, 0);
      for (int k = 0; k < this->blob_bottom_data_->shape(1); ++k) {
        if (k == label_value) {
          Dtype neuron_value = this->blob_bottom_data_->data_at(i, k, 0, 0);
          objs_std_dev_[k] +=
              (neuron_value - objs_mean_[k]) * (neuron_value - objs_mean_[k]);
        } else {
          Dtype neuron_value = this->blob_bottom_data_->data_at(i, k, 0, 0);
          not_objs_std_dev_[k] += (neuron_value - not_objs_mean_[k]) *
                                  (neuron_value - not_objs_mean_[k]);
        }
      }
    }

    for (int i = 0; i < objs_mean_.size(); ++i) {
      int normalizer = objs_count_[i];
      if (normalizer == 0) {
        objs_std_dev_[i] = static_cast<Dtype>(0);
      } else {
        objs_std_dev_[i] = objs_std_dev_[i] / (static_cast<Dtype>(normalizer));
      }
      objs_std_dev_[i] = sqrt(objs_std_dev_[i]);
      //     LOG(INFO) << "Variance for positive class " << i
      //               << " is: " << objs_std_dev_[i] * objs_std_dev_[i];
    }
    for (int i = 0; i < not_objs_mean_.size(); ++i) {
      int normalizer = not_objs_count_[i];
      if (normalizer == 0) {
        not_objs_std_dev_[i] = static_cast<Dtype>(0);
      } else {
        not_objs_std_dev_[i] =
            not_objs_std_dev_[i] / (static_cast<Dtype>(normalizer));
      }
      not_objs_std_dev_[i] = sqrt(not_objs_std_dev_[i]);
      //     LOG(INFO) << "Variance for negative class " << i
      //               << " is: " << not_objs_std_dev[i] * not_objs_std_dev[i];
    }
  }

  virtual ~PSDALayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Ground truth related variables
  std::vector<Dtype> objs_mean_;
  std::vector<Dtype> objs_std_dev_;
  std::vector<Dtype> not_objs_mean_;
  std::vector<Dtype> not_objs_std_dev_;
  std::vector<Dtype> objs_count_;
  std::vector<Dtype> not_objs_count_;
  // Helper variables
  int num_labels_;
  int num_samples_;
};

TYPED_TEST_CASE(PSDALayerTest, TestDtypes);

TYPED_TEST(PSDALayerTest, TestSetup) {
  LayerParameter layer_param;
  PSDALayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < layer.blobs().size(); ++i) {
    EXPECT_EQ(layer.blobs()[0]->num(), this->num_labels_);
    EXPECT_EQ(layer.blobs()[0]->channels(), 1);
    EXPECT_EQ(layer.blobs()[0]->height(), 1);
    EXPECT_EQ(layer.blobs()[0]->width(), 1);
  }
}

TYPED_TEST(PSDALayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  PSDALayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Compute ground truth
  this->ComputeGroundTruth();
  // Check counts first
  for (int i = 0; i < this->objs_count_.size(); ++i) {
    // Test learned parameters of the layer
    EXPECT_NEAR(this->objs_mean_[i], layer.blobs()[0]->data_at(i, 0, 0, 0),
                0.001);
    EXPECT_NEAR(this->objs_std_dev_[i], layer.blobs()[1]->data_at(i, 0, 0, 0),
                0.001);
    EXPECT_NEAR(this->not_objs_mean_[i], layer.blobs()[2]->data_at(i, 0, 0, 0),
                0.001);
    EXPECT_NEAR(this->not_objs_std_dev_[i],
                layer.blobs()[3]->data_at(i, 0, 0, 0), 0.001);
    EXPECT_EQ(this->objs_count_[i], layer.blobs()[4]->data_at(i, 0, 0, 0));
  }
}

TYPED_TEST(PSDALayerTest, TestForwardBatchesCPU) {
  int num_batches = 4;
  int batch_size = (this->num_samples_) / num_batches;
  // Setup bottoms for batched data
  vector<int> shape(2);
  shape[0] = batch_size;
  shape[1] = this->num_labels_;
  Blob<TypeParam>* const blob_bottom_batch_data = new Blob<TypeParam>();
  blob_bottom_batch_data->Reshape(shape);
  Blob<TypeParam>* const blob_bottom_batch_labels = new Blob<TypeParam>();
  shape.resize(1);
  blob_bottom_batch_labels->Reshape(shape);
  vector<Blob<TypeParam>*> blob_batch_bottom_vec;
  blob_batch_bottom_vec.push_back(blob_bottom_batch_data);
  blob_batch_bottom_vec.push_back(blob_bottom_batch_labels);

  LayerParameter layer_param;
  PSDALayer<TypeParam> layer(layer_param);
  layer.SetUp(blob_batch_bottom_vec, this->blob_top_vec_);
  for (int n = 0; n < num_batches; ++n) {
    // Copy data for each batch
    const TypeParam* bottom_data = this->blob_bottom_data_->cpu_data();
    const TypeParam* label_data = this->blob_bottom_label_->cpu_data();
    TypeParam* batch_bottom_data = blob_bottom_batch_data->mutable_cpu_data();
    TypeParam* batch_label_data = blob_bottom_batch_labels->mutable_cpu_data();
    EXPECT_EQ(blob_bottom_batch_labels->count(), batch_size);
    //     LOG(INFO) << this->blob_bottom_data_->shape(0) << ", " <<
    //     this->blob_bottom_data_->shape(1);
    //     LOG(INFO) << this->blob_bottom_data_->count();
    for (int i = 0; i < blob_bottom_batch_labels->count(); ++i) {
      batch_label_data[i] = label_data[batch_size * n + i];
      for (int k = 0; k < this->num_labels_; ++k) {
        batch_bottom_data[i * this->num_labels_ + k] =
            bottom_data[(batch_size * n + i) * this->num_labels_ + k];
      }
    }
    layer.Forward(blob_batch_bottom_vec, this->blob_top_vec_);
  }
  this->ComputeGroundTruth();
  for (int i = 0; i < this->objs_count_.size(); ++i) {
    // Test learned parameters of the layer
    EXPECT_NEAR(this->objs_mean_[i], layer.blobs()[0]->data_at(i, 0, 0, 0),
                0.001);
    EXPECT_NEAR(this->objs_std_dev_[i], layer.blobs()[1]->data_at(i, 0, 0, 0),
                0.001);
    EXPECT_NEAR(this->not_objs_mean_[i], layer.blobs()[2]->data_at(i, 0, 0, 0),
                0.001);
    EXPECT_NEAR(this->not_objs_std_dev_[i],
                layer.blobs()[3]->data_at(i, 0, 0, 0), 0.001);
    EXPECT_EQ(this->objs_count_[i], layer.blobs()[4]->data_at(i, 0, 0, 0));
  }
}

}  // namespace caffe
