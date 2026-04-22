use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tdb_index::vamana::VamanaIndex;

fn bench_vamana_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("vamana_build");
    group.sample_size(10); // Build is expensive, fewer samples.

    for n in [100, 500, 1000] {
        let dim = 32;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut v = vec![0.01f32; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &vectors, |bench, vectors| {
            bench.iter(|| {
                let mut index = VamanaIndex::new(dim, 16);
                for (i, vec) in vectors.iter().enumerate() {
                    index.insert(i as u64, vec);
                }
                index.build();
                black_box(&index);
            });
        });
    }
    group.finish();
}

fn bench_vamana_query(c: &mut Criterion) {
    let dim = 32;
    let n = 1000;
    let mut index = VamanaIndex::new(dim, 16);
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut v = vec![0.01f32; dim];
            v[i % dim] = 1.0;
            v
        })
        .collect();
    for (i, vec) in vectors.iter().enumerate() {
        index.insert(i as u64, vec);
    }
    index.build();

    let query = &vectors[500];

    c.bench_function("vamana_query_1k", |bench| {
        bench.iter(|| index.query(black_box(query), 10));
    });
}

criterion_group!(benches, bench_vamana_build, bench_vamana_query);
criterion_main!(benches);
