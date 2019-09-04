#include <driver_types.h>

#include <crypto/Argon2_constants.h>

#include "../../../common/common.h"

#include "crypto/argon2_hasher/hash/Hasher.h"
#include "crypto/argon2_hasher/hash/argon2/Argon2.h"

#include "CudaHasher.h"

#define THREADS_PER_LANE                    8
#define BLOCK_SIZE_UINT4                    64
#define BLOCK_SIZE_UINT                     256
#define KERNEL_WORKGROUP_SIZE   		    32
#define ARGON2_PREHASH_DIGEST_LENGTH_UINT   16
#define ARGON2_PREHASH_SEED_LENGTH_UINT     18


#include "blake2b.cu"

#define COMPUTE(alo, ahi, blo, bhi, clo, chi, dlo, dhi)	\
	asm ("{"	\
		".reg .u32 s1, s2, s3, s4;\n\t"	\
		"mul.lo.u32 s3, %0, %2;\n\t"	\
		"mul.hi.u32 s4, %0, %2;\n\t"	\
		"add.cc.u32 s3, s3, s3;\n\t"	\
		"addc.u32 s4, s4, s4;\n\t"	\
		"add.cc.u32 s1, %0, %2;\n\t"	\
		"addc.u32 s2, %1, %3;\n\t"	\
		"add.cc.u32 %0, s1, s3;\n\t"	\
		"addc.u32 %1, s2, s4;\n\t"	\
		"xor.b32 s1, %0, %6;\n\t"	\
		"xor.b32 %6, %1, %7;\n\t"	\
		"mov.b32 %7, s1;\n\t"	\
		"mul.lo.u32 s3, %4, %6;\n\t"	\
		"mul.hi.u32 s4, %4, %6;\n\t"	\
		"add.cc.u32 s3, s3, s3;\n\t"	\
		"addc.u32 s4, s4, s4;\n\t"	\
		"add.cc.u32 s1, %4, %6;\n\t"	\
		"addc.u32 s2, %5, %7;\n\t"	\
		"add.cc.u32 %4, s1, s3;\n\t"	\
		"addc.u32 %5, s2, s4;\n\t"	\
		"xor.b32 s3, %2, %4;\n\t"	\
		"xor.b32 s4, %3, %5;\n\t"	\
		"shf.r.wrap.b32 %3, s4, s3, 24;\n\t"	\
		"shf.r.wrap.b32 %2, s3, s4, 24;\n\t"	\
		"mul.lo.u32 s3, %0, %2;\n\t"	\
		"mul.hi.u32 s4, %0, %2;\n\t"	\
		"add.cc.u32 s3, s3, s3;\n\t"	\
		"addc.u32 s4, s4, s4;\n\t"	\
		"add.cc.u32 s1, %0, %2;\n\t"	\
		"addc.u32 s2, %1, %3;\n\t"	\
		"add.cc.u32 %0, s1, s3;\n\t"	\
		"addc.u32 %1, s2, s4;\n\t"	\
		"xor.b32 s3, %0, %6;\n\t"	\
		"xor.b32 s4, %1, %7;\n\t"	\
		"shf.r.wrap.b32 %7, s4, s3, 16;\n\t"	\
		"shf.r.wrap.b32 %6, s3, s4, 16;\n\t"	\
		"mul.lo.u32 s3, %4, %6;\n\t"	\
		"mul.hi.u32 s4, %4, %6;\n\t"	\
		"add.cc.u32 s3, s3, s3;\n\t"	\
		"addc.u32 s4, s4, s4;\n\t"	\
		"add.cc.u32 s1, %4, %6;\n\t"	\
		"addc.u32 s2, %5, %7;\n\t"	\
		"add.cc.u32 %4, s1, s3;\n\t"	\
		"addc.u32 %5, s2, s4;\n\t"	\
		"xor.b32 s3, %2, %4;\n\t"	\
		"xor.b32 s4, %3, %5;\n\t"	\
		"shf.r.wrap.b32 %3, s3, s4, 31;\n\t"	\
		"shf.r.wrap.b32 %2, s4, s3, 31;\n\t"	\
	"}" : "+r"(alo), "+r"(ahi), "+r"(blo), "+r"(bhi), "+r"(clo), "+r"(chi), "+r"(dlo), "+r"(dhi));

#define G1()           \
{                           \
    COMPUTE(data_a.x, data_a.y, data_c.x, data_c.y, data_e.x, data_e.y, data_g.x, data_g.y) \
    COMPUTE(data_a.z, data_a.w, data_c.z, data_c.w, data_e.z, data_e.w, data_g.z, data_g.w) \
    COMPUTE(data_b.x, data_b.y, data_d.x, data_d.y, data_f.x, data_f.y, data_h.x, data_h.y) \
    COMPUTE(data_b.z, data_b.w, data_d.z, data_d.w, data_f.z, data_f.w, data_h.z, data_h.w) \
}

#define G2()           \
{                           \
    COMPUTE(data_a.x, data_a.y, data_c.z, data_c.w, data_f.x, data_f.y, data_h.z, data_h.w) \
    COMPUTE(data_a.z, data_a.w, data_d.x, data_d.y, data_f.z, data_f.w, data_g.x, data_g.y) \
    COMPUTE(data_b.x, data_b.y, data_d.z, data_d.w, data_e.x, data_e.y, data_g.z, data_g.w) \
    COMPUTE(data_b.z, data_b.w, data_c.x, data_c.y, data_e.z, data_e.w, data_h.x, data_h.y) \
}

#define SHUFFLE() \
{           \
    local_mem[id] = data_a; \
    local_mem[id + 8] = data_b; \
    local_mem[id + 16] = data_c; \
    local_mem[id + 24] = data_d; \
    local_mem[id + 32] = data_e; \
    local_mem[id + 40] = data_f; \
    local_mem[id + 48] = data_g; \
    local_mem[id + 56] = data_h; \
    __syncwarp(); \
    data_a = local_mem[id * 8]; \
    data_b = local_mem[id * 8 + 1]; \
    data_c = local_mem[id * 8 + 2]; \
    data_d = local_mem[id * 8 + 3]; \
    data_e = local_mem[id * 8 + 4]; \
    data_f = local_mem[id * 8 + 5]; \
    data_g = local_mem[id * 8 + 6]; \
    data_h = local_mem[id * 8 + 7]; \
}

inline __host__ __device__ void operator^=( uint4& a, uint4 s) {
   a.x ^= s.x; a.y ^= s.y; a.z ^= s.z; a.w ^= s.w;
}

__global__ void fill_blocks(uint32_t *scratchpad0,
							uint32_t *scratchpad1,
							uint32_t *scratchpad2,
							uint32_t *scratchpad3,
							uint32_t *scratchpad4,
							uint32_t *scratchpad5,
							uint32_t *out,
                            uint32_t *refs, // 32 bit
                            uint32_t *idxs, // first bit is keep flag, next 31 bit is current idx
							uint32_t *segments,
							int memsize,
							int lanes,
                            int seg_length,
                            int seg_count,
							int threads_per_chunk,
							int thread_idx) {
    extern __shared__ uint32_t shared[];
    uint4 data_a, data_b, data_c, data_d, data_e, data_f, data_g, data_h;
    uint4 saved_a, saved_b, saved_c, saved_d, saved_e, saved_f, saved_g, saved_h;

    int session = threadIdx.x / THREADS_PER_LANE;
    int id = threadIdx.x % THREADS_PER_LANE;
    int wave_id = threadIdx.x % 32;

    int local_hash = session % 4;
    int lane = session / 4; // 4 hashes session for each lane
    int base_hash = (blockIdx.x * 4);
    int mem_hash = base_hash + thread_idx;

	int lane_length = seg_length * 4;

    uint4 *local_mem = reinterpret_cast<uint4*>(shared + (lane + local_hash * lanes) * BLOCK_SIZE_UINT);
    uint32_t *local_refs = shared + lanes * 4 * BLOCK_SIZE_UINT + lane * 32;
    uint32_t *local_idxs = shared + lanes * 4 * BLOCK_SIZE_UINT + (lanes + lane) * 32;

    int scratchpad_location = mem_hash / threads_per_chunk;
    uint4 *memory = reinterpret_cast<uint4*>(scratchpad0);
    if(scratchpad_location == 1) memory = reinterpret_cast<uint4*>(scratchpad1);
    if(scratchpad_location == 2) memory = reinterpret_cast<uint4*>(scratchpad2);
    if(scratchpad_location == 3) memory = reinterpret_cast<uint4*>(scratchpad3);
    if(scratchpad_location == 4) memory = reinterpret_cast<uint4*>(scratchpad4);
    if(scratchpad_location == 5) memory = reinterpret_cast<uint4*>(scratchpad5);
    int hash_offset = mem_hash - scratchpad_location * threads_per_chunk;
    memory = memory + hash_offset * (memsize >> 4); // memsize / 16 -> 16 bytes in uint4

	uint4 *next_block;
	uint4 *prev_block;
	uint4 *ref_block;
    uint32_t *seg_refs, *seg_idxs;

    segments += (lane * 3);

	for(int s = 0; s < (seg_count / lanes); s++) {
		int idx = ((s == 0) ? 2 : 0); // index for first slice in each lane is 2
		int with_xor = ((s >= 4) ? 1 : 0);
		int keep = 1;
		int slice = s % 4;
		int pass = s / 4;

		uint32_t *cur_seg = &segments[s * lanes * 3];

		uint32_t cur_idx = cur_seg[0];
        uint32_t prev_idx = cur_seg[1];
        uint32_t seg_type = cur_seg[2];
        uint32_t ref_idx = 0;

        prev_block = memory + prev_idx * BLOCK_SIZE_UINT4 * 4; // 4 hashes are intercalated in a single block

        data_a = prev_block[wave_id];
        data_b = prev_block[wave_id + 32];
        data_c = prev_block[wave_id + 64];
        data_d = prev_block[wave_id + 96];
        data_e = prev_block[wave_id + 128];
        data_f = prev_block[wave_id + 160];
        data_g = prev_block[wave_id + 192];
        data_h = prev_block[wave_id + 224];

        __syncthreads();

        if(seg_type == 0) {
            seg_refs = refs + ((s * lanes + lane) * seg_length - ((s > 0) ? lanes : lane) * 2);
            if(idxs != NULL) seg_idxs = idxs + ((s * lanes + lane) * seg_length - ((s > 0) ? lanes : lane) * 2);

            for (cur_idx--;idx < seg_length; seg_refs += 32, seg_idxs += 32) {
				uint64_t i_limit = seg_length - idx;
				if (i_limit > 32) i_limit = 32;

                local_refs[wave_id] = seg_refs[wave_id];

                if(idxs != NULL) {
                    local_idxs[wave_id] = seg_idxs[wave_id];
                }

                for (int i = 0; i < i_limit; i++, idx++) {
                    ref_idx = local_refs[i];

                    if(idxs != NULL) {
                        cur_idx = local_idxs[i];
                        keep = cur_idx & 0x80000000;
                        cur_idx = cur_idx & 0x7FFFFFFF;
                    }
                    else
                        cur_idx++;

                    ref_block = memory + ref_idx * BLOCK_SIZE_UINT4 * 4;
                    next_block = memory + cur_idx * BLOCK_SIZE_UINT4 * 4;

                    data_a ^= ref_block[wave_id];
                    data_b ^= ref_block[wave_id + 32];
                    data_c ^= ref_block[wave_id + 64];
                    data_d ^= ref_block[wave_id + 96];
                    data_e ^= ref_block[wave_id + 128];
                    data_f ^= ref_block[wave_id + 160];
                    data_g ^= ref_block[wave_id + 192];
                    data_h ^= ref_block[wave_id + 224];

                    saved_a = data_a;
                    saved_b = data_b;
                    saved_c = data_c;
                    saved_d = data_d;
                    saved_e = data_e;
                    saved_f = data_f;
                    saved_g = data_g;
                    saved_h = data_h;

					G1();
                    G2();
                    SHUFFLE();
                    G1();
                    G2();
                    SHUFFLE();

                    if(with_xor == 1) {
                        saved_a ^= next_block[wave_id];
                        saved_b ^= next_block[wave_id + 32];
                        saved_c ^= next_block[wave_id + 64];
                        saved_d ^= next_block[wave_id + 96];
                        saved_e ^= next_block[wave_id + 128];
                        saved_f ^= next_block[wave_id + 160];
                        saved_g ^= next_block[wave_id + 192];
                        saved_h ^= next_block[wave_id + 224];
                    }

                    data_a ^= saved_a;
                    data_b ^= saved_b;
                    data_c ^= saved_c;
                    data_d ^= saved_d;
                    data_e ^= saved_e;
                    data_f ^= saved_f;
                    data_g ^= saved_g;
                    data_h ^= saved_h;

                    if(keep > 0) {
                        next_block[wave_id] = data_a;
                        next_block[wave_id + 32] = data_b;
                        next_block[wave_id + 64] = data_c;
                        next_block[wave_id + 96] = data_d;
                        next_block[wave_id + 128] = data_e;
                        next_block[wave_id + 160] = data_f;
                        next_block[wave_id + 192] = data_g;
                        next_block[wave_id + 224] = data_h;
					}
                }
            }
        }
        else {
            for (; idx < seg_length; idx++, cur_idx++) {
				uint32_t pseudo_rand_lo = __shfl_sync(0xffffffff, data_a.x, local_hash * 8);
				uint32_t pseudo_rand_hi = __shfl_sync(0xffffffff, data_a.y, local_hash * 8);

				uint64_t ref_lane = pseudo_rand_hi % lanes; // thr_cost
				uint32_t reference_area_size = 0;
				if(pass > 0) {
					if (lane == ref_lane) {
						reference_area_size = lane_length - seg_length + idx - 1;
					} else {
						reference_area_size = lane_length - seg_length + ((idx == 0) ? (-1) : 0);
					}
				}
				else {
					if (lane == ref_lane) {
						reference_area_size = slice * seg_length + idx - 1; // seg_length
					} else {
						reference_area_size = slice * seg_length + ((idx == 0) ? (-1) : 0);
					}
				}
				asm("{mul.hi.u32 %0, %1, %1; mul.hi.u32 %0, %0, %2; }": "=r"(pseudo_rand_lo) : "r"(pseudo_rand_lo), "r"(reference_area_size));

				uint32_t relative_position = reference_area_size - 1 - pseudo_rand_lo;

				ref_idx = ref_lane * lane_length + (((pass > 0 && slice < 3) ? ((slice + 1) * seg_length) : 0) + relative_position) % lane_length;

				ref_block = memory + ref_idx * BLOCK_SIZE_UINT4 * 4;
                next_block = memory + cur_idx * BLOCK_SIZE_UINT4 * 4;

                data_a ^= ref_block[wave_id];
                data_b ^= ref_block[wave_id + 32];
                data_c ^= ref_block[wave_id + 64];
                data_d ^= ref_block[wave_id + 96];
                data_e ^= ref_block[wave_id + 128];
                data_f ^= ref_block[wave_id + 160];
                data_g ^= ref_block[wave_id + 192];
                data_h ^= ref_block[wave_id + 224];

                saved_a = data_a;
                saved_b = data_b;
                saved_c = data_c;
                saved_d = data_d;
                saved_e = data_e;
                saved_f = data_f;
                saved_g = data_g;
                saved_h = data_h;

                G1();
                G2();
                SHUFFLE();
                G1();
                G2();
                SHUFFLE();

                if(with_xor == 1) {
                    saved_a ^= next_block[wave_id];
                    saved_b ^= next_block[wave_id + 32];
                    saved_c ^= next_block[wave_id + 64];
                    saved_d ^= next_block[wave_id + 96];
                    saved_e ^= next_block[wave_id + 128];
                    saved_f ^= next_block[wave_id + 160];
                    saved_g ^= next_block[wave_id + 192];
                    saved_h ^= next_block[wave_id + 224];
                }

                data_a ^= saved_a;
                data_b ^= saved_b;
                data_c ^= saved_c;
                data_d ^= saved_d;
                data_e ^= saved_e;
                data_f ^= saved_f;
                data_g ^= saved_g;
                data_h ^= saved_h;

                next_block[wave_id] = data_a;
                next_block[wave_id + 32] = data_b;
                next_block[wave_id + 64] = data_c;
                next_block[wave_id + 96] = data_d;
                next_block[wave_id + 128] = data_e;
                next_block[wave_id + 160] = data_f;
                next_block[wave_id + 192] = data_g;
                next_block[wave_id + 224] = data_h;
            }
        }
	}

    local_mem[id * 8] = data_a;
    local_mem[id * 8 + 1] = data_b;
    local_mem[id * 8 + 2] = data_c;
    local_mem[id * 8 + 3] = data_d;
    local_mem[id * 8 + 4] = data_e;
    local_mem[id * 8 + 5] = data_f;
    local_mem[id * 8 + 6] = data_g;
    local_mem[id * 8 + 7] = data_h;

    __syncthreads();

	// at this point local_mem will contain the final blocks

	if(lane == 0) { // first lane needs to acumulate results
        data_a = make_uint4(0, 0, 0, 0);
        data_b = make_uint4(0, 0, 0, 0);
        data_c = make_uint4(0, 0, 0, 0);
        data_d = make_uint4(0, 0, 0, 0);
        data_e = make_uint4(0, 0, 0, 0);
        data_f = make_uint4(0, 0, 0, 0);
        data_g = make_uint4(0, 0, 0, 0);
        data_h = make_uint4(0, 0, 0, 0);

        local_mem = reinterpret_cast<uint4*>(shared + local_hash * lanes * BLOCK_SIZE_UINT);
		for(int l=0; l<lanes; l++) {
			uint4 *block = local_mem + l * BLOCK_SIZE_UINT4;
            data_a ^= block[id * 8];
            data_b ^= block[id * 8 + 1];
            data_c ^= block[id * 8 + 2];
            data_d ^= block[id * 8 + 3];
            data_e ^= block[id * 8 + 4];
            data_f ^= block[id * 8 + 5];
            data_g ^= block[id * 8 + 6];
            data_h ^= block[id * 8 + 7];
		}

		uint4 *out_mem = reinterpret_cast<uint4*>(out + (base_hash + local_hash) * BLOCK_SIZE_UINT);
        out_mem[id * 8] = data_a;
        out_mem[id * 8 + 1] = data_b;
        out_mem[id * 8 + 2] = data_c;
        out_mem[id * 8 + 3] = data_d;
        out_mem[id * 8 + 4] = data_e;
        out_mem[id * 8 + 5] = data_f;
        out_mem[id * 8 + 6] = data_g;
        out_mem[id * 8 + 7] = data_h;
	}
};

__global__ void prehash (uint32_t *scratchpad0,
                        uint32_t *scratchpad1,
                        uint32_t *scratchpad2,
                        uint32_t *scratchpad3,
                        uint32_t *scratchpad4,
                        uint32_t *scratchpad5,
                        uint32_t *preseed,
                        int memsize,
                        int memcost,
                        int lanes,
                        int passes,
                        int pwdlen,
                        int saltlen,
                        int seg_length,
                        int threads,
                        int threads_per_chunk,
                        int thread_idx) { // len is given in uint32 units
    extern __shared__ uint32_t shared[]; // size = max(lanes * 2, 8) * 88

	int seeds_batch_size = blockDim.x / 4; // number of seeds per block
	int hash_batch_size = seeds_batch_size / (lanes * 2); // number of hashes per block

	int id = threadIdx.x; // minimum 32 threads
	int thr_id = id % 4; // thread id in session
	int session = id / 4; // blake2b hashing session

    int hash_base = blockIdx.x * hash_batch_size;
    int hash_idx = session / (lanes * 2);

    if((hash_base + hash_idx) < threads) {
        int hash_session = session % (lanes * 2); // session in hash

        int lane = hash_session / 2;  // 2 lanes
        int idx = hash_session % 2; // idx in lane

        uint32_t *local_outBuff = &shared[session * BLOCK_SIZE_UINT];
        uint32_t *local_mem = &shared[seeds_batch_size * BLOCK_SIZE_UINT + session * BLAKE_SHARED_MEM_UINT];

        uint64_t *h = (uint64_t *) &local_mem[20];
        uint32_t *buf = (uint32_t *) &h[10];
        uint32_t *value = &buf[32];
        uint32_t *local_preseed = &value[1];

        uint32_t *cursor_in = preseed;
        uint32_t *cursor_out = local_preseed;

        for(int i=0; i < (pwdlen >> 2); i++, cursor_in += 4, cursor_out += 4) {
            cursor_out[thr_id] = cursor_in[thr_id];
        }

        if(thr_id == 0) {
            for (int i = 0; i < (pwdlen % 4); i++) {
                cursor_out[i] = cursor_in[i];
            }

            uint32_t nonce = (preseed[9] >> 24) | (preseed[10] << 8);
            nonce += (hash_base + hash_idx);
            local_preseed[9] = (preseed[9] & 0x00FFFFFF) | (nonce << 24);
            local_preseed[10] = (preseed[10] & 0xFF000000) | (nonce >> 8);
        }

        int buf_len = blake2b_init(h, ARGON2_PREHASH_DIGEST_LENGTH_UINT, thr_id);
        *value = lanes; //lanes
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = 32; //outlen
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = memcost; //m_cost
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = passes; //t_cost
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = ARGON2_VERSION; //version
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = ARGON2_TYPE_VALUE; //type
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = pwdlen * 4; //pw_len
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        buf_len = blake2b_update(local_preseed, pwdlen, h, buf, buf_len, thr_id);
        *value = saltlen * 4; //salt_len
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
		buf_len = blake2b_update(local_preseed, saltlen, h, buf, buf_len, thr_id);
        *value = 0; //secret_len
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        buf_len = blake2b_update(NULL, 0, h, buf, buf_len, thr_id);
        *value = 0; //ad_len
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        buf_len = blake2b_update(NULL, 0, h, buf, buf_len, thr_id);

        blake2b_final(local_mem, ARGON2_PREHASH_DIGEST_LENGTH_UINT, h, buf, buf_len, thr_id);

        if (thr_id == 0) {
            local_mem[ARGON2_PREHASH_DIGEST_LENGTH_UINT] = idx;
            local_mem[ARGON2_PREHASH_DIGEST_LENGTH_UINT + 1] = lane;
        }

        blake2b_digestLong(local_outBuff, ARGON2_DWORDS_IN_BLOCK, local_mem, ARGON2_PREHASH_SEED_LENGTH_UINT, thr_id,
            &local_mem[20]);

        int mem_hash = hash_base + thread_idx;
        int scratchpad_location = mem_hash / threads_per_chunk;
        uint4 *memory = reinterpret_cast<uint4*>(scratchpad0);
        if(scratchpad_location == 1) memory = reinterpret_cast<uint4*>(scratchpad1);
        if(scratchpad_location == 2) memory = reinterpret_cast<uint4*>(scratchpad2);
        if(scratchpad_location == 3) memory = reinterpret_cast<uint4*>(scratchpad3);
        if(scratchpad_location == 4) memory = reinterpret_cast<uint4*>(scratchpad4);
        if(scratchpad_location == 5) memory = reinterpret_cast<uint4*>(scratchpad5);
        int hash_offset = mem_hash - scratchpad_location * threads_per_chunk;
        memory = memory + hash_offset * (memsize >> 4); // memsize / 16 -> 16 bytes in uint4

        int lane_length = seg_length * 4;

        uint32_t *mem_seed = shared + hash_idx * lanes * 2 * BLOCK_SIZE_UINT;
        uint4 *seed_dst = memory + lane * (lane_length * 4) * BLOCK_SIZE_UINT4; // lane_length * 4 because we intercalate 4 hashes in memory
        uint4 *seed_src = reinterpret_cast<uint4*>(mem_seed + lane * 2 * BLOCK_SIZE_UINT);

        int thr_in_lane = threadIdx.x % THREADS_PER_LANE;

        for(int i=0; i < 8; i++)
        seed_dst[id + i * 32] = seed_src[i + thr_in_lane * 8]; // id * 8 - split the block in 8 succesive regions of 8 uint4 each

        seed_src += BLOCK_SIZE_UINT4;
        seed_dst += (4 * BLOCK_SIZE_UINT4);

        for(int i=0; i < 8; i++)
        seed_dst[id + i * 32] = seed_src[i + thr_in_lane * 8];
    }
}

__global__ void posthash (
        uint32_t *hash,
        uint32_t *out,
        uint32_t *preseed) {
    extern __shared__ uint32_t shared[]; // size = 120

    int hash_id = blockIdx.x;
    int thread = threadIdx.x;

    uint32_t *local_hash = hash + hash_id * ((ARGON2_RAW_LENGTH / 4) + 1);
    uint32_t *local_out = out + hash_id * BLOCK_SIZE_UINT;

    blake2b_digestLong(local_hash, ARGON2_RAW_LENGTH / 4, local_out, ARGON2_DWORDS_IN_BLOCK, thread, shared);

    if(thread == 0) {
        uint32_t nonce = (preseed[9] >> 24) | (preseed[10] << 8);
        nonce += hash_id;
        local_hash[ARGON2_RAW_LENGTH / 4] = nonce;
    }
}

void cuda_allocate(CudaDeviceInfo *device, double chunks, size_t chunk_size) {
	Argon2Profile *profile = device->profileInfo.profile;

	device->error = cudaSetDevice(device->cudaIndex);
	if(device->error != cudaSuccess) {
		device->errorMessage = "Error setting current device for memory allocation.";
		return;
	}

	size_t allocated_mem_for_current_chunk = 0;

	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_0, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_1, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_2, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_3, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_4, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_5, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}

	uint32_t *refs = (uint32_t *)malloc(profile->blockRefsSize * sizeof(uint32_t));
	for(int i=0;i<profile->blockRefsSize;i++) {
		refs[i] = profile->blockRefs[i*3 + 1];
	}

	device->error = cudaMalloc(&device->arguments.refs, profile->blockRefsSize * sizeof(uint32_t));
	if(device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}

	device->error = cudaMemcpy(device->arguments.refs, refs, profile->blockRefsSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->errorMessage = "Error copying memory.";
		return;
	}
	free(refs);

	if(profile->succesiveIdxs == 1) {
		device->arguments.idxs = NULL;
	}
	else {
		uint32_t *idxs = (uint32_t *) malloc(profile->blockRefsSize * sizeof(uint32_t));
		for (int i = 0; i < profile->blockRefsSize; i++) {
			idxs[i] = profile->blockRefs[i * 3];
			if (profile->blockRefs[i * 3 + 2] == 1) {
				idxs[i] |= 0x80000000;
			}
		}

		device->error = cudaMalloc(&device->arguments.idxs, profile->blockRefsSize * sizeof(uint32_t));
		if (device->error != cudaSuccess) {
			device->errorMessage = "Error allocating memory.";
			return;
		}

		device->error = cudaMemcpy(device->arguments.idxs, idxs, profile->blockRefsSize * sizeof(uint32_t),
								   cudaMemcpyHostToDevice);
		if (device->error != cudaSuccess) {
			device->errorMessage = "Error copying memory.";
			return;
		}
		free(idxs);
	}

	//reorganize segments data
	device->error = cudaMalloc(&device->arguments.segments, profile->segCount * 3 * sizeof(uint32_t));
	if(device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.segments, profile->segments, profile->segCount * 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->errorMessage = "Error copying memory.";
		return;
	}

#ifdef PARALLEL_CUDA
	int threads = device->profileInfo.threads / 2;
#else
	int threads = device->profileInfo.threads;
#endif

	size_t preseed_memory_size = profile->pwdLen * 4;
	size_t seed_memory_size = threads * (profile->thrCost * 2) * ARGON2_BLOCK_SIZE;
	size_t out_memory_size = threads * ARGON2_BLOCK_SIZE;
	size_t hash_memory_size = threads * (xmrig::ARGON2_HASHLEN + 4);

    device->error = cudaMalloc(&device->arguments.preseedMemory[0], preseed_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.seedMemory[0], seed_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.outMemory[0], out_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.hashMemory[0], hash_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMallocHost(&device->arguments.hostSeedMemory[0], 132 * threads);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating pinned memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.preseedMemory[1], preseed_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.seedMemory[1], seed_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.outMemory[1], out_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.hashMemory[1], hash_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMallocHost(&device->arguments.hostSeedMemory[1], 132 * threads);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating pinned memory.";
        return;
    }
}

void cuda_free(CudaDeviceInfo *device) {
	cudaSetDevice(device->cudaIndex);

	if(device->arguments.idxs != NULL) {
		cudaFree(device->arguments.idxs);
		device->arguments.idxs = NULL;
	}

	if(device->arguments.refs != NULL) {
		cudaFree(device->arguments.refs);
		device->arguments.refs = NULL;
	}

	if(device->arguments.segments != NULL) {
		cudaFree(device->arguments.segments);
		device->arguments.segments = NULL;
	}

    if(device->arguments.memoryChunk_0 != NULL) {
        cudaFree(device->arguments.memoryChunk_0);
        device->arguments.memoryChunk_0 = NULL;
    }

    if(device->arguments.memoryChunk_1 != NULL) {
        cudaFree(device->arguments.memoryChunk_1);
        device->arguments.memoryChunk_1 = NULL;
    }

    if(device->arguments.memoryChunk_2 != NULL) {
        cudaFree(device->arguments.memoryChunk_2);
        device->arguments.memoryChunk_2 = NULL;
    }

    if(device->arguments.memoryChunk_3 != NULL) {
        cudaFree(device->arguments.memoryChunk_3);
        device->arguments.memoryChunk_3 = NULL;
    }

    if(device->arguments.memoryChunk_4 != NULL) {
        cudaFree(device->arguments.memoryChunk_4);
        device->arguments.memoryChunk_4 = NULL;
    }

    if(device->arguments.memoryChunk_5 != NULL) {
        cudaFree(device->arguments.memoryChunk_5);
        device->arguments.memoryChunk_5 = NULL;
    }

    if(device->arguments.preseedMemory != NULL) {
        for(int i=0;i<2;i++) {
            if(device->arguments.preseedMemory[i] != NULL)
                cudaFree(device->arguments.preseedMemory[i]);
            device->arguments.preseedMemory[i] = NULL;
        }
    }

	if(device->arguments.seedMemory != NULL) {
		for(int i=0;i<2;i++) {
			if(device->arguments.seedMemory[i] != NULL)
				cudaFree(device->arguments.seedMemory[i]);
			device->arguments.seedMemory[i] = NULL;
		}
	}

	if(device->arguments.outMemory != NULL) {
		for(int i=0;i<2;i++) {
			if(device->arguments.outMemory[i] != NULL)
				cudaFree(device->arguments.outMemory[i]);
			device->arguments.outMemory[i] = NULL;
		}
	}

    if(device->arguments.hashMemory != NULL) {
        for(int i=0;i<2;i++) {
            if(device->arguments.hashMemory[i] != NULL)
                cudaFree(device->arguments.hashMemory[i]);
            device->arguments.hashMemory[i] = NULL;
        }
    }

	if(device->arguments.hostSeedMemory != NULL) {
		for(int i=0;i<2;i++) {
			if(device->arguments.hostSeedMemory[i] != NULL)
				cudaFreeHost(device->arguments.hostSeedMemory[i]);
			device->arguments.hostSeedMemory[i] = NULL;
		}
	}

	cudaDeviceReset();
}

bool cuda_kernel_prehasher(void *memory, int threads, Argon2Profile *profile, void *user_data) {
    CudaGpuMgmtThreadData *gpumgmt_thread = (CudaGpuMgmtThreadData *)user_data;
    CudaDeviceInfo *device = gpumgmt_thread->device;
    cudaStream_t stream = (cudaStream_t)gpumgmt_thread->deviceData;

    int sessions = max(profile->thrCost * 2, (uint32_t)8);
    double hashes_per_block = sessions / (profile->thrCost * 2.0);
    size_t work_items = sessions * 4;

    gpumgmt_thread->lock();

    memcpy(device->arguments.hostSeedMemory[gpumgmt_thread->threadId], memory, gpumgmt_thread->hashData.inSize);

    device->error = cudaMemcpyAsync(device->arguments.preseedMemory[gpumgmt_thread->threadId], device->arguments.hostSeedMemory[gpumgmt_thread->threadId], gpumgmt_thread->hashData.inSize, cudaMemcpyHostToDevice, stream);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error writing to gpu memory.";
        gpumgmt_thread->unlock();
        return false;
    }

	prehash <<< ceil(threads / hashes_per_block), work_items, sessions * (BLAKE_SHARED_MEM + ARGON2_BLOCK_SIZE), stream>>> (
            (uint32_t*)device->arguments.memoryChunk_0,
            (uint32_t*)device->arguments.memoryChunk_1,
            (uint32_t*)device->arguments.memoryChunk_2,
            (uint32_t*)device->arguments.memoryChunk_3,
            (uint32_t*)device->arguments.memoryChunk_4,
            (uint32_t*)device->arguments.memoryChunk_5,
			device->arguments.preseedMemory[gpumgmt_thread->threadId],
            profile->memSize,
            profile->memCost,
			profile->thrCost,
			profile->segCount / (4 * profile->thrCost),
            gpumgmt_thread->hashData.inSize / 4,
			profile->saltLen,
            profile->segSize,
            threads,
            device->profileInfo.threads_per_chunk,
            gpumgmt_thread->threadsIdx);

    return true;
}

void *cuda_kernel_filler(int threads, Argon2Profile *profile, void *user_data) {
	CudaGpuMgmtThreadData *gpumgmt_thread = (CudaGpuMgmtThreadData *)user_data;
	CudaDeviceInfo *device = gpumgmt_thread->device;
	cudaStream_t stream = (cudaStream_t)gpumgmt_thread->deviceData;

    size_t work_items = KERNEL_WORKGROUP_SIZE * profile->thrCost;
    size_t shared_mem = profile->thrCost * (4 * ARGON2_BLOCK_SIZE + 128 + (profile->succesiveIdxs == 1 ? 128 : 0));

	fill_blocks <<<threads / 4, work_items, shared_mem, stream>>> ((uint32_t*)device->arguments.memoryChunk_0,
			(uint32_t*)device->arguments.memoryChunk_1,
			(uint32_t*)device->arguments.memoryChunk_2,
			(uint32_t*)device->arguments.memoryChunk_3,
			(uint32_t*)device->arguments.memoryChunk_4,
			(uint32_t*)device->arguments.memoryChunk_5,
			device->arguments.outMemory[gpumgmt_thread->threadId],
			device->arguments.refs,
			device->arguments.idxs,
			device->arguments.segments,
			profile->memSize,
			profile->thrCost,
			profile->segSize,
			profile->segCount,
			device->profileInfo.threads_per_chunk,
            gpumgmt_thread->threadsIdx);

	return (void *)1;
}

bool cuda_kernel_posthasher(void *memory, int threads, Argon2Profile *profile, void *user_data) {
	CudaGpuMgmtThreadData *gpumgmt_thread = (CudaGpuMgmtThreadData *)user_data;
	CudaDeviceInfo *device = gpumgmt_thread->device;
	cudaStream_t stream = (cudaStream_t)gpumgmt_thread->deviceData;

    size_t work_items = 4;

	posthash <<<threads, work_items, BLAKE_SHARED_MEM, stream>>> (
            device->arguments.hashMemory[gpumgmt_thread->threadId],
            device->arguments.outMemory[gpumgmt_thread->threadId],
            device->arguments.preseedMemory[gpumgmt_thread->threadId]);

	device->error = cudaMemcpyAsync(device->arguments.hostSeedMemory[gpumgmt_thread->threadId], device->arguments.hashMemory[gpumgmt_thread->threadId], threads * (xmrig::ARGON2_HASHLEN + 4), cudaMemcpyDeviceToHost, stream);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error reading gpu memory.";
		gpumgmt_thread->unlock();
		return false;
	}

	while(cudaStreamQuery(stream) != cudaSuccess) {
		this_thread::sleep_for(chrono::milliseconds(10));
		continue;
	}

    memcpy(memory, device->arguments.hostSeedMemory[gpumgmt_thread->threadId], threads * (xmrig::ARGON2_HASHLEN + 4));
	gpumgmt_thread->unlock();

	return memory;
}