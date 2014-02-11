__kernel void gram(__global const float* data,
                   __global float* gram,
                   uint cols,
                   const char* type
                   float sigma)
{
    uint row1 = get_global_id(0);
    uint row2 = get_global_id(1);
    uint rows = get_global_size(0);
    if (type[0] == 'g')
    {
        float t = 0, s;
        for (uint n = 0; n < cols; n++)
        {
            s = data[cols * row1 + n] - data[cols * row2 + n];
            t += s * s;
        }
        gram[row1 * rows + row2] = gram[row2 * rows + row1] = exp(-.5*sigma*t);
    }
    else if (strcmp(type, "linear") == 0)
    {
        float t = 0;
        for (uint n = 0; n < cols; n++)
            t += data[cols * row1 + n] * data[cols * row2 + n];
        gram[row1 * rows + row2] = gram[row2 * rows + row1] = t;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}
