	?d??~K5@?d??~K5@!?d??~K5@      ??!       "q
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
	K ?)???K ?)???!K ?)???      ??!       "	???0?@???0?@!???0?@*      ??!       2	?lt?Oq|??lt?Oq|?!?lt?Oq|?:	{Cr29,@{Cr29,@!{Cr29,@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qr?U%?R@y8ީj??;@