use ort::session::{Session, builder::GraphOptimizationLevel};

fn main() {
    let model = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(10)
        .unwrap()
        .commit_from_file("models/gigaam_ctc/v3_e2e_ctc.onnx")
        .unwrap();
    dbg!(&model);
}
