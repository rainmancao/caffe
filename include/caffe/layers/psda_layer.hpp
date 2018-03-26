#ifndef CAFFE_PSDA_LAYER_HPP_
#define CAFFE_PSDA_LAYER_HPP_

#include <vector>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

// #include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the PSDA parameters of the training dataset
 */
template <typename Dtype>
class PSDALayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides PSDAParameter psda_param,
   *     with PSDALayer options:
   */
  explicit PSDALayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PSDA"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  // Blobs for positive class mean, positive class standard deviation,
  // negative class mean, negative class standard deviation and counts
  virtual inline int ExactNumTopBlobs() const { return 1; }
  //   virtual inline int MinTopBlobs() const { return 5; }
  //   virtual inline int MaxTopBlos() const { return 5; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$, a Blob with values in
   *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
   *      the @f$ K = CHW @f$ classes. Each @f$ x_n @f$ is mapped to a predicted
   *      label @f$ \hat{l}_n @f$ given by its maximal index:
   *      @f$ \hat{l}_n = \arg\max\limits_k x_{nk} @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels @f$ l @f$, an integer-valued Blob with values
   *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
   *      indicating the correct class label among the @f$ K @f$ classes
   * @param top output Blob vector (length K)
   *   -# To be implemented
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- PSDALayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) {
        NOT_IMPLEMENTED;
      }
    }
  }

  int label_axis_, outer_num_, inner_num_;

  // Number of iterations
  int iter_count_;
  // Epoch length specified in number of iterations
  int epoch_length_;
  // Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  // Accumulators for positive category statistics
  std::vector<boost::accumulators::accumulator_set<
      Dtype,
      boost::accumulators::features<boost::accumulators::tag::mean,
                                    boost::accumulators::tag::variance> > >
      objs_acc_;
  // Accumulators for negative category statistics
  std::vector<boost::accumulators::accumulator_set<
      Dtype,
      boost::accumulators::features<boost::accumulators::tag::mean,
                                    boost::accumulators::tag::variance> > >
      not_objs_acc_;
  // Number of categories
  int num_categories_;
};

}  // namespace caffe

#endif  // CAFFE_PSDA_LAYER_HPP_
