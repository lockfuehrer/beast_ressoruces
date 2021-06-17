int main(int argc, char *argv[]) {
  listing();
  
}

void listing(){

// loop over rows
for (int i=0; i<mlen-sublen; i++){

  // loop over columns
  for (int j=i+sublen; j<mlen; j++){ 
    if (i!=0) // except for the first row

      // streaming dot product - Pearson's correlation (firstpart)
      QT[j-i-sublen] += df[i] * dg[j] + df[j] * dg[i];

    // streaming dot product - scaling (second part)
    double cr = QT[j-i-sublen] * norm[i] * norm[j];

    //row-wise argmin
    if(cr > mp[i]){ 
      mp[i]=cr; 
      mpi[i]=j;
    }

    //column -wise argmin
    if(cr > mp[j]){ 
      mp[j]=cr; 
      mpi[j]=i;
    }
