__kernel void conj_grad(int dim, int num_vals, __local float *r, 
      __local float *x, __local float* A_times_p, __local float *p,
      __global int *rows, __global int *cols, __global float *A, 
      __global float *b, __global float *result) {

   local float alpha, r_length, old_r_dot_r, new_r_dot_r;
   local int iteration;

   int id = get_local_id(0);//id从0开始往下执行,id数量是行数量
   int start_index = -1;
   int end_index = -1;
   float Ap_dot_p;

   /* Find matrix values for each work-item */
   for(int i=id; i<num_vals; i++) {
      if((rows[i] == id) && (start_index == -1)) 
         start_index = i;
      else if((rows[i] == id+1) && (end_index == -1))  {
         end_index = i-1;
         break;
      }
      else if((i == num_vals-1) && (end_index == -1)) {
         end_index = i;
      }
   }//矩阵第一列是111 222这样的形式，A矩阵每一行是一个id，一个工作项
   //从id到id+1，实际上id就增加了一个，但是i从前一行到后一行（从111到222），每个id包含了一行
   //eg id=0,start-end:-1 - -1
   //   id=1,start-end:1-4
   //   id=2,start-end:5-9

   /* Set the initial guess, residual, and direction */
   x[id] = 0.0f;
   r[id] = b[id];
   p[id] = b[id];
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Compute old r_dot_r */
   if(id == 0) {
      old_r_dot_r = 0.0f;
      for(int i=0; i<dim; i++) {
         old_r_dot_r += r[i] * r[i];
      }
      r_length = sqrt(old_r_dot_r);
   }//只有在id=0的时候执行，id=1,2，...，num_row都不执行，因为这是初始化
   barrier(CLK_LOCAL_MEM_FENCE);

   iteration = 0;
   while((iteration < 1000) && (r_length >= 0.01)) {

      /* Compute Ap.p */
      A_times_p[id] = 0.0f;//each work item（id）处理的是A矩阵一行的数据
      for(int i=start_index; i<=end_index; i++) {
         A_times_p[id] += A[i] * p[cols[i]];
      }//A[i]是矩阵具体值，cols[i]是列的索引，比如在id为1时，start-end: 1-4,我们传进去i=1（排在第一个的值）
      //但是输出可能是cols[1]=3，p[3]的位置才能正确对应,其他位置A都是0，没有计算的必要
      barrier(CLK_LOCAL_MEM_FENCE);

      /* Compute alpha = r.r/Ap.p */
      if(id == 0) {
         Ap_dot_p = 0.0f;
         for(int i=0; i<dim; i++) {
            Ap_dot_p += A_times_p[i] * p[i];
         } 
         alpha = old_r_dot_r/Ap_dot_p;
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      /* Compute the next guess and the next residual */
      x[id] += alpha * p[id];
      r[id] -= alpha * A_times_p[id];
      barrier(CLK_LOCAL_MEM_FENCE);

      /* Compute new r_dot_r */
      if(id == 0) {
         new_r_dot_r = 0.0f;
         for(int i=0; i<dim; i++) {
            new_r_dot_r += r[i] * r[i];
         }
         r_length = sqrt(new_r_dot_r);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      /* Compute new direction: p = r + (new_r_dot_r/old_r_dot_r) * p */
      p[id] = r[id] + (new_r_dot_r/old_r_dot_r) * p[id];
      barrier(CLK_LOCAL_MEM_FENCE);

      old_r_dot_r = new_r_dot_r;

      if(id==0) {
        iteration++;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   result[0] = iteration * 1.0f;
   result[1] = r_length;
}

   
