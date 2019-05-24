
#if 0
int main() {
	if (g_test_params.run) {
		test_env_parse();
	}

	(void)g_cupti_subscriber;
	(void)g_cupti_runtime_cbids;

	_thread_main = pthread_self();

	int threads = 1024;
	
	cupti_benchmark_start(&g_event_data);
	
	clock64_t* thread_times = zallocNN(sizeof(thread_times[0]) * threads);

	while (g_event_data.count_event_groups_read
				 < g_event_data.num_event_groups) {
		
		gpu_test_matrix_vec_mul(threads, thread_times);
		CUDA_RUNTIME_FN(cudaDeviceSynchronize());
	}
	
	for (int i = 0; i < threads; ++i) {
		printf("[%i] time: %llu\n", i, thread_times[i]);
	}

	cupti_report_event_data(&g_event_data);
	
	cupti_benchmark_end();

	cleanup();
	
	return 0;
}
#endif
