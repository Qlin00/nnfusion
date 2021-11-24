void MatrixMulCUDA_8bit_bias(float *input0, float *input1, float *input2, float *input3, float *input4, float *input5, float * input6, float *output0) 
{
    const unsigned int M_GLOBAL=M_VALUE;
    const unsigned int K_GLOBAL=K_VALUE;
    const unsigned int N_GLOBAL=N_VALUE;

    const uint8_t * A = reinterpret_cast<uint8_t*>(input0); // activation
    const uint8_t * B_val =  reinterpret_cast<uint8_t*>(input1); // weight
    const int * B_row = reinterpret_cast< int *>(input2);
    const int * B_col = reinterpret_cast< int *>(input3);
    const int alpha = (int)(*input4);
    const int integer = (int)(*input5);
    int * C = reinterpret_cast< int *>(input6);
    uint8_t * D = reinterpret_cast<uint8_t*>(output0);

}