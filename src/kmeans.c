void kmeans ( int dim_num, int point_num, int cluster_num, int it_max,
  int *it_num, double point[], int cluster[], double cluster_center[],
  int cluster_population[], double cluster_energy[] )
/*
  R
*/
{
  int ci, cj, i, j, k, swap;
  double *distsq;

  assign_to_clusters(dim_num, point_num, cluster_num, point, cluster, cluster_center, cluster_population);
/*
  Average the points in each cluster to get a new cluster center.
*/
  r8vec_zeros ( dim_num * cluster_num, cluster_center );
  for ( j = 0; j < point_num; j++ )
  {
    k = cluster[j];
    for ( i = 0; i < dim_num; i++ )
    {
      cluster_center[i+k*dim_num] = cluster_center[i+k*dim_num]
        + point[i+j*dim_num];
    }
  }
  for ( k = 0; k < cluster_num; k++ )
  {
    for ( i = 0; i < dim_num; i++ )
    {
      cluster_center[i+k*dim_num] = cluster_center[i+k*dim_num]
        / ( double ) ( cluster_population[k] );
    }
  }
/*
  Carry out the iteration.
*/
  *it_num = 0;
  distsq = ( double * ) malloc ( cluster_num * sizeof ( double ) );
  while ( *it_num < it_max )
  {
    *it_num = *it_num + 1;
    swap = 0;
    for ( j = 0; j < point_num; j++ )
    {
      ci = cluster[j];
      if ( cluster_population[ci] <= 1 )
      {
        continue;
      }
      for ( cj = 0; cj < cluster_num; cj++ )
      {
        if ( cj == ci )
        {
          distsq[cj] = 0.0;
          for ( i = 0; i < dim_num; i++ )
          {
            distsq[cj] = distsq[cj]
              + pow ( point[i+j*dim_num] - cluster_center[i+cj*dim_num], 2 );
          }
          distsq[cj] = distsq[cj] * ( double ) ( cluster_population[cj] )
            / ( double ) ( cluster_population[cj] - 1 );
        }
        else if ( cluster_population[cj] == 0 )
        {
          for ( i = 0; i < dim_num; i++ )
          {
            cluster_center[i+cj*dim_num] = point[i+j*dim_num];
          }
          distsq[cj] = 0.0;
        }
        else
        {
          distsq[cj] = 0.0;
          for ( i = 0; i < dim_num; i++ )
          {
            distsq[cj] = distsq[cj]
              + pow ( point[i+j*dim_num] - cluster_center[i+cj*dim_num], 2 );
          }
          distsq[cj] = distsq[cj] * ( double ) ( cluster_population[cj] )
            / ( double ) ( cluster_population[cj] + 1 );
        }
      }
/*
  Find the index of the minimum value of DISTSQ.
*/
      k = r8vec_min_index ( cluster_num, distsq );
      swap = reassign_point(swap, k, ci, cj, dim_num, point, cluster, cluster_center, cluster_population);
    }
/*
  Exit if no reassignments were made during this iteration.
*/
    if ( swap == 0 )
    {
      break;
    }
  }
  free ( distsq );
  return;
}

void assign_to_clusters(int dim_num, int point_num, int cluster_num,
  double point[], int cluster[], double cluster_center[],
  int cluster_population[]){
    /*
  Assign each point to the nearest cluster center.
    */
    int j;
    double dist_point, min_dist_point;
  for ( j = 0; j < point_num; j++ )
  {
    min_dist_point = r8_huge ( );
    cluster[j] = -1;
    for ( k = 0; k < cluster_num; k++ )
    {
      dist_point = 0.0;
      for ( i = 0; i < dim_num; i++ )
      {
        dist_point = dist_point +
          pow ( point[i+j*dim_num] - cluster_center[i+k*dim_num], 2 );
      }
      if ( dist_point < min_dist_point )
      {
        min_dist_point = dist_point;
        cluster[j] = k;
        cluster_population[k] = cluster_population[k] + 1;
      }
    }
  }
}

int reassign_point(int swap,int k, int ci, int cj, int dim_num, double point[], int cluster[], 
  double cluster_center[], int cluster_population[]){
      int i;
    /*
       if not classified to correct cluster, reassign point
    */
      if ( k == ci )
      {
        continue;
      }
      cj = k;
      for ( i = 0; i < dim_num; i++ )
      {
        cluster_center[i+ci*dim_num] = ( ( double ) ( cluster_population[ci] )
          * cluster_center[i+ci*dim_num] - point[i+j*dim_num] )
          / ( double ) ( cluster_population[ci] - 1 );

        cluster_center[i+cj*dim_num] = ( ( double ) ( cluster_population[cj] )
          * cluster_center[i+cj*dim_num] + point[i+j*dim_num] )
          / ( double ) ( cluster_population[cj] + 1 );
      }
      cluster_population[ci] = cluster_population[ci] - 1;
      cluster_population[cj] = cluster_population[cj] + 1;
      cluster[j] = cj;
      swap = swap + 1;
      return swap;
}