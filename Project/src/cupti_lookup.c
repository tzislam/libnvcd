#include "cupti_lookup.h"
#include <string.h>
#include <inttypes.h>
#include "util.h"
#include "list.h"

/*
 * List of event strings as listed in the CUPTI event documentation
 * for CUDA toolkit v9.2
 */

const char* g_cupti_event_names_2x[NUM_CUPTI_EVENTS_2X] = {
	/* 
	 * domain_a 
   */

	"sm_cta_launched",
	
	// 6
	"l1_local_load_hit",
	"l1_local_load_miss",
	"l1_local_store_hit",
	"l1_local_store_miss",
	"l1_global_load_hit",
	"l1_global_load_miss",

	// 1
	"uncached_global_load_transaction",

	// 1
	"global_store_transaction",

	// 1
	"l1_shared_bank_conflict",

	// 2
	"tex0_cache_sector_queries",
	"tex0_cache_sector_misses",

	// 2
	"tex1_cache_sector_queries",
	"tex1_cache_sector_misses",

	/* 
	 * domain_b
   */
	
	// 14
	"l2_subp0_write_sector_misses",
	"l2_subp1_write_sector_misses",
	"l2_subp0_read_sector_misses",
	"l2_subp1_read_sector_misses",
	"l2_subp0_write_sector_queries",
	"l2_subp1_write_sector_queries",
	"l2_subp0_read_sector_queries",
	"l2_subp1_read_sector_queries",
	"l2_subp0_read_hit_sectors",
	"l2_subp1_read_hit_sectors",
	"l2_subp0_read_tex_sector_queries",
	"l2_subp1_read_tex_sector_queries",
	"l2_subp0_read_tex_hit_sectors",
	"l2_subp1_read_tex_hit_sectors",

	// 4
	"fb_subp0_read_sectors",
	"fb_subp1_read_sectors",
	"fb_subp0_write_sectors",
	"fb_subp1_write_sectors",

	// 4
	"fb0_subp0_read_sectors",
	"fb0_subp1_read_sectors",
	"fb0_subp0_write_sectors",
	"fb0_subp1_write_sectors",

	// 4
	"fb1_subp0_read_sectors",
	"fb1_subp1_read_sectors",
	"fb1_subp0_write_sectors",
	"fb1_subp1_write_sectors",

	/*
	 * domain_c
	 */

	"gld_inst_8bit",
	"gld_inst_16bit",
	"gld_inst_32bit",
	"gld_inst_64bit",
	"gld_inst_128bit",

	"gst_inst_8bit",
	"gst_inst_16bit",
	"gst_inst_32bit",
	"gst_inst_64bit",
	"gst_inst_128bit",

	/*
	 * domain_d
	 */

	"branch",
	"divergent_branch",
	"warps_launched",
	"threads_launched",
	"active_warps",
	"active_cycles",

	"local_load",
	"local_store",
	"gld_request",
	"gst_request",
	"shared_load",
	"shared_store",
	"prof_trigger_XX",

	"inst_issued",
	"inst_issued1_0",
	"inst_issued2_0",
	"inst_issued1_1",
	"inst_issued2_1",
	"inst_executed",
	
	"thread_inst_executed_0",
	"thread_inst_executed_1"
};

/*
 * List of event metrics as listed in the CUPTI event documentation
 * for CUDA toolkit v9.2
 */

const char* g_cupti_metrics_3x[NUM_CUPTI_METRICS_3X] = {

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

typedef struct cupti_name_map {
	list_t self;
	char* name;
	CUpti_EventID id;
} cupti_name_map_t;

static cupti_name_map_t* g_name_map_list = NULL;

static void cupti_name_map_push(cupti_name_map_t* node) {
	list_push_fn_impl(&g_name_map_list,
										node,
										cupti_name_map_t,
										self);
}

static void cupti_name_map_free_node(cupti_name_map_t* n) {
	free(n->name);
	n->name = NULL;
}

void cupti_name_map_free() {
	list_free_fn_impl(g_name_map_list,
										cupti_name_map_t,
										cupti_name_map_free_node,
										self);
}

void cupti_map_event_name_to_id(const char* event_name, CUpti_EventID event_id) {
	if (cupti_find_event_name_from_id(event_id) == NULL) {
		cupti_name_map_t* node = mallocNN(sizeof(cupti_name_map_t));

		node->name = strdup(event_name);

		NOT_NULL(node->name);
		
		node->id = event_id;
		node->self.next = NULL;

		cupti_name_map_push(node);
	}
}

const char* cupti_find_event_name_from_id(CUpti_EventID id) {
	const char* ret = NULL;

	list_t* n = list_node(g_name_map_list,
												cupti_name_map_t,
												self);

	while (n != NULL && ret == NULL) {
		cupti_name_map_t* nm = list_base(n,
																		 cupti_name_map_t,
																		 self);
		if (nm->id == id) {
			ret = nm->name;
		}
		
		n = n->next;
	}

	return ret;
}

static char _peg_buffer[1 << 13] = { 0 };

static void print_event_group(cupti_event_data_t* e, uint32_t group) {
	// used for iterative bounds checking
#define peg_buffer_length (sizeof(_peg_buffer) / sizeof(_peg_buffer[0])) - 1
	
	memset(&_peg_buffer[0], 0, sizeof(_peg_buffer));
	
	uint64_t* pcounters = &e->event_counter_buffer[0];
	
	uint32_t ib_offset = e->event_id_buffer_offsets[group];
	uint32_t cb_offset = e->event_counter_buffer_offsets[group];
	
  uint32_t nepg = e->num_events_per_group[group];
  uint32_t nipg = e->num_instances_per_group[group];

	uint32_t next_cb_offset = 0;
	uint32_t next_ib_offset = 0;

	int ptr = 0;
	
	// bounds check ordering for
	// event_counter_buffer_offsets
	{
		uint32_t prev_cb_offset = (group > 0) ?
			
			e->event_counter_buffer_offsets[group - 1] :
			0;

		uint32_t prev_cb_offset_add = (group > 0) ?

			(e->num_events_per_group[group - 1] *
			 e->num_instances_per_group[group - 1]) :
			0;

		ASSERT(prev_cb_offset + prev_cb_offset_add == cb_offset);
	}

	// bounds check ordering for
	// event_id_buffer_offsets
	{
		uint32_t prev_ib_offset = (group > 0) ?

			e->event_id_buffer_offsets[group - 1] :
			0;

		uint32_t prev_ib_offset_add = (group > 0) ?
			
			e->num_events_per_group[group - 1] :
			0;

		ASSERT(prev_ib_offset + prev_ib_offset_add == ib_offset);
	}

	// used for iterative bounds checking
	{
		next_cb_offset =
			group < (e->num_event_groups - 1) ?
			e->event_counter_buffer_offsets[group + 1] :
			e->event_counter_buffer_length;
	}
	
	// used for iterative bounds checking
	{
		next_ib_offset =
			group < (e->num_event_groups - 1) ?
			e->event_id_buffer_offsets[group + 1] :
			e->event_id_buffer_length;
	}
	
	for (uint32_t i = 0; i < nepg; ++i) {
		ASSERT(ib_offset + i < next_ib_offset);
		ASSERT(ptr < peg_buffer_length);

		CUpti_EventID eid = e->event_id_buffer[ib_offset + i];
		
		{
			const char* name = cupti_find_event_name_from_id(eid);
			
			ptr += sprintf(&_peg_buffer[ptr],
										 "[%" PRIu32 " (eid: 0x%" PRIx32 ")] %s:\n",
										 i,
										 eid,
										 name);
		}
		
		for (uint32_t j = 0; j < nipg; ++j) {
		  uint32_t k = cb_offset + j * nepg + i;

			ASSERT(k < next_cb_offset);
			
			ptr += sprintf(&_peg_buffer[ptr],
										 "\t[%" PRIu32 "] %" PRIu64 " | 0x%" PRIx64 "\n",
										 j,
										 pcounters[k],
										 pcounters[k]);
		}
	}

	ASSERT(ptr <= peg_buffer_length);
	
	printf("======GROUP %" PRIu32  "=======\n"
				 "%s"
				 "===\n",
				 group,
				 &_peg_buffer[0]);

	#undef peg_buffer_length
}

void cupti_report_event_data(cupti_event_data_t* e) {
	for (uint32_t i = 0; i < e->num_event_groups; ++i) {
		print_event_group(e, i);
	}
}
