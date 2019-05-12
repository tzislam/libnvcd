#include "cupti_lookup.h"

/*
 * List of event strings as listed in the CUPTI event documentation
 * for CUDA toolkit v9.2
 */

const char* g_cupti_events_3x[NUM_CUPTI_EVENTS_3X] = {

	// 6
	"achieved_occupancy",
	"alu_fu_utilization",
	"atomic_replay_overhead",
	"atomic_throughput",
  "atomic_transactions",
  "atomic_transactions_per_request",

	// 1
  "branch_efficiency",

	// 3
	"cf_executed",
	"cf_fu_utilization",
	"cf_issued",

	// 5
	"dram_read_throughput", 
	"dram_read_transactions",
	"dram_utilization", 	
	"dram_write_throughput",
	"dram_write_transactions",

	// 3
	"ecc_throughput",
	"ecc_transactions",
	"eligible_warps_per_cycle",

	// 11
	"flop_count_dp", 
	"flop_count_dp_add",
	"flop_count_dp_fma",
	"flop_count_dp_mul",
	"flop_count_sp", 
	"flop_count_sp_add",
	"flop_count_sp_fma",
	"flop_count_sp_mul",
	"flop_count_sp_special",
	"flop_dp_efficiency",
	"flop_sp_efficiency",

	// 12
	"gld_efficiency",
	"gld_requested_throughput",
	"gld_throughput",
	"gld_transactions",
	"gld_transactions_per_request",
	"global_cache_replay_overhead",
	"global_replay_overhead",
	"gst_efficiency",
	"gst_requested_throughput",
	"gst_throughput",
	"gst_transactions",
	"gst_transactions_per_request",

	// 17
	"inst_bit_convert",
	"inst_compute_ld_st",
	"inst_control",
	"inst_executed",
	"inst_fp_32",
	"inst_fp_64",
	"inst_integer",
	"inst_inter_thread_communication",
	"inst_issued",
	"inst_misc",
	"inst_per_warp",
	"inst_replay_overhead",
	"ipc",
	"ipc_instance",
	"issue_slot_utilization",
	"issue_slots",
	"issued_ipc",

	// 29
	"l1_cache_global_hit_rate",
	"l1_cache_local_hit_rate",
	"l1_shared_utilization",
	"l2_atomic_throughput",
	"l2_atomic_transactions",
	"l2_l1_read_hit_rate",
	"l2_l1_read_throughput",
	"l2_l1_read_transactions",
	"l2_l1_write_throughput",
	"l2_l1_write_transactions",
	"l2_read_throughput",
	"l2_read_transactions",
	"l2_tex_read_transactions",
	"l2_tex_read_hit_rate",
	"l2_tex_read_throughput",
	"l2_utilization", 	
	"l2_write_throughput",
	"l2_write_transactions",
	"ldst_executed",
	"ldst_fu_utilization",
	"ldst_issued",
	"local_load_throughput",
	"local_load_transactions",
	"local_load_transactions_per_request",
	"local_memory_overhead",
	"local_replay_overhead",
	"local_store_throughput",
	"local_store_transactions",
	"local_store_transactions_per_request",

	// 6
	"nc_cache_global_hit_rate",
	"nc_gld_efficiency",
	"nc_gld_requested_throughput",
	"nc_gld_throughput",
	"nc_l2_read_throughput",
	"nc_l2_read_transactions",

	// 27 
	"shared_efficiency",
	"shared_load_throughput",
	"shared_load_transactions",
	"shared_load_transactions_per_request",
	"shared_replay_overhead",
	"shared_store_throughput",
	"shared_store_transactions",
	"shared_store_transactions_per_request",
	"sm_efficiency",
	"sm_efficiency_instance",
	"stall_constant_memory_dependency",
	"stall_exec_dependency",
	"stall_inst_fetch",
	"stall_memory_dependency",
	"stall_memory_throttle",
	"stall_not_selected",
	"stall_other",
	"stall_pipe_busy",
	"stall_sync",
	"stall_texture",
	"sysmem_read_throughput",
	"sysmem_read_transactions",
	"sysmem_read_utilization",
	"sysmem_utilization",
	"sysmem_write_throughput",
	"sysmem_write_transactions",
	"sysmem_write_utilization",

	// 5
	"tex_cache_hit_rate",
	"tex_cache_throughput",
	"tex_cache_transactions",
	"tex_fu_utilization",
	"tex_utilization",

	// 2
	"warp_execution_efficiency",
	"warp_nonpred_execution_efficiency"
};
