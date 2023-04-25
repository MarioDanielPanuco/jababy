use clap::Parser;
use llm_chain::{chains::sequential::Chain, prompt};
use llm_chain_openai::chatgpt::{Executor, Model, Step};
use std::fs;
use std::path::PathBuf;
use tracing::{info, instrument};
use tracing_subscriber;
use tracing_subscriber::fmt::format;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Set the professor's expertise
    #[arg(short, long, required(true))]
    expertise: String,

    /// Set the file containing the question
    #[arg(short, long, required(true), value_name = "FILE")]
    question_file: PathBuf,
}

#[tokio::main(flavor = "current_thread")]
#[instrument]
async fn main() {
    let cli = Cli::parse();

    // Extract the command-line arguments
    let expertise = cli.expertise;
    let question_file = cli.question_file;

    // Set up the tracing subscriber with a simple configuration
    tracing_subscriber::fmt()
        .pretty()
        .with_ansi(true)
        .init();

    info!("Expertise: {}", expertise);
    info!("Question file: {:?}", question_file);

    // Create a new ChatGPT executor with the default settings
    let exec = Executor::new_default();
    let turby: Model = Model::ChatGPT3_5Turbo;

    // Read the content of the file at "./data/questions/q1.txt"
    let question_text =
        fs::read_to_string(question_file).expect("Failed to read the file specified question file");

    // Create a chain for the Professor's prompt
    let professor_chain = Chain::new(vec![
        Step::for_prompt(
            prompt!("System: PHD in Computer Science, expertise in {{expertise}}, Respond in a combination of text and latex code blocks, or wrap math typesets within $$",
                "Make lecture notes to address this question. Provide proof if question asks for proof{{question}}")
        ), 

        Step::for_prompt(
            prompt!("Review lecture notes, make sure everything is styled in latex. Go step by and step and make sure lecture notes
                    are right, make sense, and naturally build inutition for the topi")
        )
    ]);

    // Create a chain for the TA's prompt
    let ta_chain = Chain::new(vec![
        Step::for_prompt(
            prompt!("System: You are the TA, try to implement some of lecture into Python code "),

        ),

        Step::for_prompt(prompt!("Lecture: {{latex}}\n==\n")),
    ]);

    info!("⛓️ Created Chains ⛓️");
    // Run the Professor's chain with the provided parameters
    let professor_output = professor_chain
        .run(
            vec![
                ("expertise", expertise.as_str()),
                ("question", question_text.as_str()),
            ]
            .into(),
            &exec,
        )
        .await
        .expect("Failed to run professor");

    // Write the professor's LaTeX notes to a file at "./data/responses/professor_notes.txt"
    fs::write(
        "./data/responses/professor_notes.txt",
        professor_output.to_string(),
    )
    .expect("Failed to write the professor's LaTeX notes to ./data/responses/professor_notes.txt");

    println!("Prof: \n===\n{:?}", professor_output.to_string());

    let ta_output = ta_chain
        .run(
            vec![("latex", professor_output.to_string().as_str())].into(),
            &exec,
        )
        .await
        .expect("Failed to run TA");

    // Write the TA's code to a file at "./data/responses/ta_code.py"
    fs::write("./data/responses/ta_code.py", ta_output.to_string())
        .expect("Failed to write the TA's code to ./data/responses/ta_code.py");

    // Print the result to the console
    println!("TA: \n===\n{:?}", professor_output.to_string());
}
