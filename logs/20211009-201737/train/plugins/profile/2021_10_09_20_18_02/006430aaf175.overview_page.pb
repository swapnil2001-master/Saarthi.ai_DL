?	?d??~K5@?d??~K5@!?d??~K5@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?d??~K5@K ?)???1???0?@A?lt?Oq|?I{Cr29,@r0*	???S?My@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???	/???!?~e@?F@)?K?e?%??1????|:9@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?A%?c\??![lf??4@)?A%?c\??1[lf??4@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?$y????!R???$ER@)?~?٭e??1p?????1@:Preprocessing2E
Iterator::Root????????!??uem?:@)??p?5??1??????0@:Preprocessing2T
Iterator::Root::ParallelMapV2A?]??a??!8P??c?$@)A?]??a??18P??c?$@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????!j?\`@)????1j?\`@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea??+e??!?s?1ev!@)ˀ??,'??1?}???@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?_=?[???!??@?%?"@)?Ǚ&l?i?1??$\??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?66.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIr?U%?R@Q8ީj??;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	K ?)???K ?)???!K ?)???      ??!       "	???0?@???0?@!???0?@*      ??!       2	?lt?Oq|??lt?Oq|?!?lt?Oq|?:	{Cr29,@{Cr29,@!{Cr29,@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qr?U%?R@y8ީj??;@?"?
?model/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/EncoderDNN/EmbeddingLookup/EmbeddingLookupUnique/UniqueUnique?Hs????!?Hs????"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop?????l??!??Z????0"&
CudnnRNNCudnnRNN䴻????!RLH??"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam?L??'??!'|???"(

concat_1_0ConcatV2??Џ?T??!?)?????"?
isparse_categorical_crossentropy_1/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits?V??4??!O?M???"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits?t????!_?q0X???" 
splitSplit_?0V??!Ҿ??	???"?
?model/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/EncoderDNN/DNN/ResidualHidden_3/dense/MatMulMatMul?,}???!r??????0"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam~?e>??!?K{V????Q      Y@Y8[?~?u>@a2)^ ?bQ@q???}>?@y??B8/??"?

both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?66.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Kepler)(: B 