/* goya_gemm -- exercise the Goya MME GEMM library.
 *   goya_gemm <busid> verify        random shapes vs CPU reference
 *   goya_gemm <busid> sweep         shape sweep across the envelope
 *   goya_gemm <busid> bench M K N R sustained throughput
 *   goya_gemm <busid> soak  SECONDS long run, random in-envelope shapes
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "goya_mme.h"

static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+1e-9*t.tv_nsec;}
static unsigned rs = 12345;
static int rnd(int n){ rs = rs*1103515245u + 12345u; return (int)((rs>>16) % (unsigned)n); }

static void ref(const int8_t*A,const int8_t*B,int32_t*C,int M,int K,int N){
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){
        int32_t a=0; for(int k=0;k<K;k++) a += (int32_t)A[m*K+k]*(int32_t)B[k*N+n];
        C[m*N+n]=a; } }

/* Runs the same GEMM REPS times. Reports worst mismatch count and whether every
   rep produced byte-identical output (deterministic) or varied (a race). */
static int  g_reps = 1;
static int  g_stable = 1;   /* set 0 if outputs differed between reps */
static long check(gmme *g,int M,int K,int N,double *gmac,int *fires){
    int8_t *A=malloc((size_t)M*K), *B=malloc((size_t)K*N);
    int32_t *C=malloc((size_t)M*N*4), *R=malloc((size_t)M*N*4);
    int32_t *P=malloc((size_t)M*N*4);
    long bad=0; double t0,t1,best=1e30;
    for(int i=0;i<M*K;i++) A[i]=(int8_t)(rnd(255)-127);
    for(int i=0;i<K*N;i++) B[i]=(int8_t)(rnd(255)-127);
    ref(A,B,R,M,K,N);
    g_stable=1;
    for(int rep=0; rep<g_reps; rep++){
        long b=0;
        memset(C,0xAB,(size_t)M*N*4);
        t0=now_s();
        if(gmme_gemm_i8(g,A,B,C,M,K,N)){ printf("    ERROR %s\n",gmme_error(g)); bad=-1; goto out; }
        t1=now_s();
        if(t1-t0 < best) best=t1-t0;
        for(int i=0;i<M*N;i++) if(C[i]!=R[i]) b++;
        if(rep==0) memcpy(P,C,(size_t)M*N*4);
        else if(memcmp(P,C,(size_t)M*N*4)) g_stable=0;
        if(b>bad) bad=b;
    }
    if(gmac) *gmac=(double)M*K*N/best/1e9;
    if(fires) *fires=gmme_fires(g);
out: free(A);free(B);free(C);free(R);free(P); return bad; }

int main(int argc,char**argv){
    const char *bus = argc>1 && strcmp(argv[1],"auto") ? argv[1] : NULL;
    const char *mode = argc>2 ? argv[2] : "verify";
    if(argc>3 && (!strcmp(mode,"sweep")||!strcmp(mode,"verify"))) g_reps=atoi(argv[3]);
    if(g_reps<1) g_reps=1;
    gmme *g = gmme_open(bus);
    if(!g || gmme_error(g)[0]){ printf("open failed: %s\n", gmme_error(g)); return 1; }
    printf("card %s  reps=%d  rows/fire: K=16->%d K=64->%d K=256->%d\n\n",
           bus?bus:"auto", g_reps, gmme_rows_per_fire(16), gmme_rows_per_fire(64), gmme_rows_per_fire(256));

    if(!strcmp(mode,"sweep")){
        int Ks[]={1,2,7,16,32,64,128,256}, Ns[]={1,3,16,64,128,256}, Ms[]={1,2,5,8,16,64,257};
        long tot=0,fail=0,unstable=0;
        printf("%-6s %-6s %-6s %-8s %-7s %-6s %-7s %s\n","M","K","N","mismatch","fires","idle","reps","GMAC/s");
        for(unsigned a=0;a<sizeof Ms/sizeof*Ms;a++)
        for(unsigned b=0;b<sizeof Ks/sizeof*Ks;b++)
        for(unsigned c=0;c<sizeof Ns/sizeof*Ns;c++){
            double gm=0; int fr=0;
            long bad=check(g,Ms[a],Ks[b],Ns[c],&gm,&fr);
            tot++; if(bad) fail++; if(!g_stable) unstable++;
            if(bad || !g_stable)
              printf("%-6d %-6d %-6d %-8ld %-7d %-6s %-7s %.2f\n",
                     Ms[a],Ks[b],Ns[c],bad,fr,gmme_idle(g)?"YES":"NO",
                     g_stable?"same":"VARIES",gm);
        }
        printf("\nSWEEP: %ld shapes x%d reps: %ld exact, %ld with mismatches, %ld non-deterministic\n",tot,g_reps,tot-fail,fail,unstable);
        gmme_close(g); return fail?1:0;
    }
    if(!strcmp(mode,"runs")){          /* exactly N gemms, count how many are wrong */
        long want=argc>3?atol(argv[3]):1000; int minK=argc>4?atoi(argv[4]):1;
        long bad=0, badelem=0, n; double t0=now_s(); int idle_ok=1;
        for(n=0;n<want;n++){
            int K=minK+rnd(GMME_MAX_K-minK+1), N=1+rnd(GMME_MAX_N), M=1+rnd(64);
            long b=check(g,M,K,N,NULL,NULL);
            if(b>0){ bad++; badelem+=b;
                printf("  FAIL #%ld  M=%-4d K=%-4d N=%-4d wrong=%ld\n",n,M,K,N,b);
                fflush(stdout); }
            if(!gmme_idle(g)) idle_ok=0;
            if((n+1)%1000==0){ printf("  ...%ld/%ld  bad=%ld  (%.0fs)\n",
                n+1,want,bad,now_s()-t0); fflush(stdout); }
        }
        printf("RUNS %ld minK=%d: %ld bad gemms (%.4f%%), %ld wrong elements, idle always=%s, %.0fs\n",
               want,minK,bad,100.0*bad/want,badelem,idle_ok?"YES":"NO",now_s()-t0);
        gmme_close(g); return bad?1:0;
    }
    if(!strcmp(mode,"stream")){   /* weights resident, activations streamed */
        int M=atoi(argv[3]),K=atoi(argv[4]),N=atoi(argv[5]);
        long R=argc>6?atol(argv[6]):100;
        int8_t *W=malloc((size_t)K*N), *X=malloc((size_t)M*K);
        int32_t*Y=malloc((size_t)M*N*4),*Rf=malloc((size_t)M*N*4);
        for(int i=0;i<K*N;i++) W[i]=(int8_t)(rnd(255)-127);
        if(gmme_load_weights(g,W,K,N)){ printf("load_weights: %s\n",gmme_error(g)); return 1; }
        printf("weights resident: K=%d N=%d (%d bytes on card)\n",
               gmme_weights_k(g), gmme_weights_n(g), K*N);
        long bad=0; double t0=now_s();
        for(long r=0;r<R;r++){
            for(int i=0;i<M*K;i++) X[i]=(int8_t)(rnd(255)-127);   /* fresh input */
            memset(Y,0xAB,(size_t)M*N*4);
            if(gmme_apply(g,X,Y,M)){ printf("apply: %s\n",gmme_error(g)); return 1; }
            ref(X,W,Rf,M,K,N);
            for(int i=0;i<M*N;i++) if(Y[i]!=Rf[i]){ bad++; break; }
        }
        double el=now_s()-t0;
        printf("streamed %ld inputs against resident weights\n",R);
        printf("  %.3f ms per apply, %.2f GMAC/s, %ld wrong, idle=%s\n",
               el/R*1e3,(double)M*K*N*R/el/1e9,bad,gmme_idle(g)?"YES":"NO");
        gmme_close(g); return bad?1:0;
    }
    if(!strcmp(mode,"one")){
        int M=atoi(argv[3]),K=atoi(argv[4]),N=atoi(argv[5]);
        g_reps = argc>6?atoi(argv[6]):10; if(g_reps<1) g_reps=1;
        double gm=0; int fr=0;
        long b=check(g,M,K,N,&gm,&fr);
        printf("M=%-5d K=%-5d N=%-5d bad=%-7ld fires=%-4d idle=%-4s %s  %.2f GMAC/s\n",
               M,K,N,b,fr,gmme_idle(g)?"YES":"NO",g_stable?"same":"VARIES",gm);
        if(b>0 && getenv("GMME_WHERE")){   /* which elements are wrong? */
            int8_t *A=malloc((size_t)M*K),*B=malloc((size_t)K*N);
            int32_t*C=malloc((size_t)M*N*4),*R=malloc((size_t)M*N*4);
            for(int i=0;i<M*K;i++)A[i]=(int8_t)(rnd(255)-127);
            for(int i=0;i<K*N;i++)B[i]=(int8_t)(rnd(255)-127);
            int shown=0;
            for(int att=0;att<60;att++){
                memset(C,0xAB,(size_t)M*N*4);
                if(gmme_gemm_i8(g,A,B,C,M,K,N)) break;
                ref(A,B,R,M,K,N);
                long n=0; int rows[512]; int cols[512]; int nr=0;
                for(int i=0;i<M*N;i++) if(C[i]!=R[i]){ if(nr<512){rows[nr]=i/N;cols[nr]=i%N;nr++;} n++; }
                if(!n) continue;
                printf("  attempt %d: %ld wrong\n",att,n);
                int rmin=rows[0],rmax=rows[0],cmin=cols[0],cmax=cols[0];
                for(int i=0;i<nr;i++){ if(rows[i]<rmin)rmin=rows[i]; if(rows[i]>rmax)rmax=rows[i];
                                       if(cols[i]<cmin)cmin=cols[i]; if(cols[i]>cmax)cmax=cols[i]; }
                { int seen[64],ns=0;
                  for(int i=0;i<nr;i++){ int dup=0;
                    for(int j=0;j<ns;j++) if(seen[j]==rows[i]) dup=1;
                    if(!dup && ns<64) seen[ns++]=rows[i]; }
                  printf("    distinct wrong rows (%d):",ns);
                  for(int j=0;j<ns&&j<20;j++) printf(" %d",seen[j]);
                  printf("\n"); }
                printf("    rows %d..%d   cols %d..%d   reread=%s\n",rmin,rmax,cmin,cmax,
                       gmme_reread_differs(g)?"DIFFERS (still settling)":"same (DRAM is wrong)");

                printf("\n"); if(++shown>=6) break; }
            free(A);free(B);free(C);free(R); }
        gmme_close(g); return b?1:0;
    }
    if(!strcmp(mode,"bench")){
        int M=argc>3?atoi(argv[3]):64, K=argc>4?atoi(argv[4]):32, N=argc>5?atoi(argv[5]):256;
        int R=argc>6?atoi(argv[6]):50;
        int8_t *A=malloc((size_t)M*K),*B=malloc((size_t)K*N);
        int32_t*C=malloc((size_t)M*N*4);
        for(int i=0;i<M*K;i++)A[i]=(int8_t)(rnd(255)-127);
        for(int i=0;i<K*N;i++)B[i]=(int8_t)(rnd(255)-127);
        if(getenv("GMME_PIN") && gmme_load_weights(g,B,K,N)){ printf("pin failed: %s\n",gmme_error(g)); return 1; }
        double t0=now_s(); long fires=0;
        for(int r=0;r<R;r++){ if(gmme_gemm_i8(g,A,B,C,M,K,N)){printf("ERR %s\n",gmme_error(g));return 1;}
                              fires+=gmme_fires(g); }
        double el=now_s()-t0;
        printf("bench M=%d K=%d N=%d x%d reps\n",M,K,N,R);
        printf("  %.1f ms total, %.3f ms/gemm, %ld fires (%.3f ms/fire)\n",
               el*1e3, el/R*1e3, fires, el/fires*1e3);
        printf("  %.2f GMAC/s   idle=%s   worst idle polls=%d\n",
               (double)M*K*N*R/el/1e9, gmme_idle(g)?"YES":"NO", gmme_idle_polls(g));
        gmme_close(g); return 0;
    }
    if(!strcmp(mode,"soak")){
        double secs=argc>3?atof(argv[3]):30.0, t0=now_s();
        int minK=argc>4?atoi(argv[4]):1;
        long runs=0,bad=0; int worstidle=1;
        while(now_s()-t0 < secs){
            int K=minK+rnd(GMME_MAX_K-minK+1), N=1+rnd(GMME_MAX_N), M=1+rnd(64);
            long b=check(g,M,K,N,NULL,NULL);
            runs++; if(b) { bad++; printf("  MISMATCH M=%d K=%d N=%d bad=%ld\n",M,K,N,b); }
            if(!gmme_idle(g)) worstidle=0;
        }
        printf("SOAK %.0fs minK=%d: %ld gemms, %ld with mismatches, idle always=%s\n",
               secs,minK,runs,bad,worstidle?"YES":"NO");
        gmme_close(g); return bad?1:0;
    }
    { long bad=0; double gm; int fr;
      int sh[][3]={{1,16,64},{4,16,64},{8,16,256},{8,32,256},{16,32,128},{64,16,256},{100,7,33}};
      printf("%-6s %-6s %-6s %-8s %-7s %-6s %-7s %s\n","M","K","N","mismatch","fires","idle","reps","GMAC/s");
      for(unsigned i=0;i<sizeof sh/sizeof*sh;i++){
        long b=check(g,sh[i][0],sh[i][1],sh[i][2],&gm,&fr); bad+=b>0?b:0;
        printf("%-6d %-6d %-6d %-8ld %-7d %-6s %-7s %.2f\n",
               sh[i][0],sh[i][1],sh[i][2],b,fr,gmme_idle(g)?"YES":"NO",
               g_stable?"same":"VARIES",gm); }
      printf("\n%s\n", bad?"FAIL":"all exact vs CPU reference");
      gmme_close(g); return bad?1:0; }
}
