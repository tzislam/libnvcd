#include "common.h"

namespace gpu {
        GPU_INL_FN void assert_cond_impl(bool_t condition, const char* message)
        {
                if (!condition) {
                        printf("DEVICE ERROR: %s\n", message);
                }
        }

#if 0
        template <typename T, sz_t N>
        class array
        {
        public:
                using value_type = T;
                using cvalue_type = const T;

                static constexpr sz_t LENGTH = N;
                
                value_type backing[N];

                GPU_FN sz_t length() const
                {
                        return N;
                }
                
                GPU_FN void fill(cvalue_type x)
                {
                        for (sz_t i = 0; i < length(); ++i) {
                                backing[i] = x;
                        }
                }
                
                GPU_FN array(cvalue_type x = value_type())
                {
                        fill(x);
                }

                GPU_FN value_type* data()
                {
                        return &backing[0];
                }

                GPU_FN value_type* data() const
                {
                        return &backing[0];
                }
                
                GPU_FN value_type& operator[] (sz_t i)
                {
                        GPU_ASSERT(i < length(), "index out of bounds");
                        return backing[i];
                }

                GPU_FN const value_type& operator[] (sz_t i) const
                {
                        GPU_ASSERT(i < length(), "index out of bounds");
                        return backing[i];
                }
        };
#endif
        

#if 0
        using char_t = char;
        
        static constexpr sz_t CHAR_MASK = (1 << (sizeof(char_t) * 8)) - 1ULL;
        
        using cstring_micro_t = array<char_t, 32>;
        using cstring_small_t = array<char_t, 64>;
        using cstring_medium_t = array<char_t, 1 << 10>;
        using cstring_large_t = array<char_t, 1 << 12>;

        using int_to_cstring_t = cstring_micro_t;

        template <typename intType>
        GPU_FN char_t to_char(const intType& x)
        {
                return static_cast<char_t>((x & CHAR_MASK) + 0x30);
        }
         
        template <typename intType>
        GPU_INL_FN int_to_cstring_t to_cstr(const intType& x)
        {
                // TODO: ensure intType is integral
                // at compile time
         
                using int_type = intType;
                
                int_to_cstring_t ret_string{};
                
                // count digits
                array<int_type, int_to_cstring_t::LENGTH - 1> digits;
                sz_t count = 0;
                {
                        int_type cx = x;

                        while (cx > 0 && count < digits.length()) {
                                int_type end = cx % 10;

                                digits[count] = end;
                                count++;

                                cx /= 10;
                        }

                        GPU_ASSERT(cx == 0, "digit count exhausted");
                }
                
                // write digits
                {
                        long_int_t ccount = static_cast<long_int_t>(count) - 1;
                        
                        while (ccount >= 0) {
                                ret_string[ccount] = to_char(digits[ccount]);
                                ccount--;
                        }
                }

                return ret_string;
        }
        #endif
        
        GPU_FN void print_thread_info()
        {
                int thread = threadIdx.x + threadIdx.y * blockDim.x;
                //int_to_cstring_t ret{to_cstr<int>(thread)};

                printf("the thread: %i\n", thread);
                //printf("Thread Num: %i, Thread String: %s\n", thread, ret.data());
        }
}

extern "C" {
        GPU_KERN_FN void gpu_kernel()
        {
                gpu::print_thread_info();
        }

        GPU_CLIENT_FN void gpu_test()
        {
                //dim3 grid(1, 1, 1);
                //dim3 block(2, 2, 1);
        
                gpu_kernel<<<1, 1>>>();
        }
}
