#include <functional>
#include <utility>
#include <vector>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/psda_layer.hpp"

namespace caffe {

template <typename Dtype>
void PSDALayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  has_ignore_label_ = this->layer_param_.psda_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.psda_param().ignore_label();
  }

  epoch_length_ = this->layer_param_.psda_param().epoch_length();
  iter_count_ = 0;
  num_categories_ = bottom[0]->shape(1);
  objs_acc_.resize(num_categories_);
  not_objs_acc_.resize(num_categories_);
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.psda_param().axis());
  bottom[0]->count(0, label_axis_);
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";

  // Create blobs to hold PSDA parameters
  int num_param_layers = 5;  // This needs to be parameterized
  this->blobs_.resize(num_param_layers);
  vector<int> weight_shape(1);
  weight_shape[0] = bottom[0]->shape(label_axis_);
  shared_ptr<ConstantFiller<Dtype> > weight_filler;
  FillerParameter filler_param;
  filler_param.set_value(Dtype(0));
  weight_filler.reset(new ConstantFiller<Dtype>(filler_param));
  for (int i = 0; i < num_param_layers; ++i) {
    this->blobs_[i].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[i].get());  // zero-fill the weights
  }
}

template <typename Dtype>
void PSDALayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    vector<int> top_shape_stats(1);
    top_shape_stats[0] = bottom[0]->shape(label_axis_);
    top[i]->Reshape(top_shape_stats);
  }
}

template <typename Dtype>
void PSDALayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

  if (this->phase_ == TRAIN) {
    const int dim = bottom[0]->count() / outer_num_;
    const int num_labels = bottom[0]->shape(label_axis_);
    CHECK_EQ(num_labels, this->num_categories_);
    Dtype* categories_count_params = this->blobs_[4]->mutable_cpu_data();
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value =
            static_cast<int>(bottom_label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        for (int k = 0; k < this->num_categories_; ++k) {
          if (k == label_value) {
            objs_acc_[k](bottom_data[i * dim + k * inner_num_ + j]);
          } else {
            not_objs_acc_[k](bottom_data[i * dim + k * inner_num_ + j]);
          }
        }
      }
    }

    Dtype* positives_mean_params = this->blobs_[0]->mutable_cpu_data();
    Dtype* positives_std_dev_params = this->blobs_[1]->mutable_cpu_data();
    Dtype* negatives_mean_params = this->blobs_[2]->mutable_cpu_data();
    Dtype* negatives_std_dev_params = this->blobs_[3]->mutable_cpu_data();
    for (int i = 0; i < this->num_categories_; ++i) {
      // Stats for positive class
      if (boost::accumulators::count(objs_acc_[i]) > 0) {
        positives_mean_params[i] = boost::accumulators::mean(objs_acc_[i]);
        positives_std_dev_params[i] =
            sqrt(boost::accumulators::variance(objs_acc_[i]));
        categories_count_params[i] = boost::accumulators::count(objs_acc_[i]);
      }
      // Stats for negative class
      if (boost::accumulators::count(not_objs_acc_[i]) > 0) {
        negatives_mean_params[i] = boost::accumulators::mean(not_objs_acc_[i]);
        negatives_std_dev_params[i] =
            sqrt(boost::accumulators::variance(not_objs_acc_[i]));
      }
    }

    iter_count_++;
    if (epoch_length_ > 0 && (iter_count_ % epoch_length_ == 0)) {
      LOG(INFO) << "Done with " << iter_count_
                << " iterations. Resetting iteration count";
      objs_acc_.clear();
      not_objs_acc_.clear();
      objs_acc_.resize(num_categories_);
      not_objs_acc_.resize(num_categories_);
    }
  } else {
    LOG(INFO) << "PSDA layer behavior during test phase under construction";
  }
  // PSDA layer should not be used as a loss function.
}

INSTANTIATE_CLASS(PSDALayer);
REGISTER_LAYER_CLASS(PSDA);

}  // namespace caffe
