use llm_chain::{chains::sequential::Chain, prompt};
use llm_chain_openai::chatgpt::{Model, Executor, Step};
use std::fs;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Create a new ChatGPT executor with the default settings
    let exec = Executor::new_default();
    let turby: Model = Model::ChatGPT3_5Turbo;

    // Read the content of the file at "./data/questions/q1.txt"
    let question_text = fs::read_to_string("./data/questions/q1.txt")
        .expect("Failed to read the file at ./data/questions/q1.txt");

    // Create a chain for the Professor's prompt
    let professor_chain = Chain::new(vec![
        Step::for_prompt(
            prompt!("System: PHD in Computer Science, expertise in {{expertise}}, Respond in a combination of text and latex code blocks, or wrap math typesets within $$",
                "Make lecture notes to address this question. Provide proof if question asks for proof{{question}}")
        )
    ]);

    // Create a chain for the TA's prompt
    let ta_chain = Chain::new(vec![
        Step::for_prompt(
            prompt!( "System: You are the TA, try to implement some of lecture into Python code (networks and numpy) and only respond with the code",
                "Lecture: {{latex}}\n==\n")
        )
    ]);

    println!("⛓️ Created Chains ⛓️\n\n");
    // Run the Professor's chain with the provided parameters
    let professor_output = professor_chain
        .run(
            vec![("expertise", "spectral graph theory"),
                 ("question", question_text.as_str())].into(),
            &exec,
        )
        .await
        .expect("Failed to run professor");

    // Write the professor's LaTeX notes to a file at "./data/responses/professor_notes.txt"
    fs::write("./data/responses/professor_notes.txt", professor_output.to_string())
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
