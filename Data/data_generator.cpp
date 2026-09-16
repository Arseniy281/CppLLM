#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

inline std::filesystem::path GetProjectRoot() {
    namespace fs = std::filesystem;
    fs::path current = fs::current_path();

    if (fs::exists(current / "Data")) {
        return current;
    }

    if (fs::exists(current.parent_path() / "Data")) {
        return current.parent_path();
    }

    throw std::runtime_error("Cannot find project root");
}

const size_t DATASET_SIZE = 20000;
const unsigned int TRAIN_SEED = 42;

inline const std::filesystem::path PROJECT_ROOT = GetProjectRoot();
inline const std::filesystem::path OUTPUT_PATH = PROJECT_ROOT / "Data/data.txt";

int RandomInt(std::mt19937& gen, int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

template <typename T> const T& RandomElement(std::mt19937& gen, 
        const std::vector<T>& values) {
    return values[RandomInt(gen, 0, (int)(values.size()) - 1)];
}

struct Fact {
    std::string question;
    std::string answer;
};

const std::vector<Fact> FACTS = {

    {"What is the capital of France?", "Paris."},
    {"What is the capital of Germany?", "Berlin."},
    {"What is the capital of Italy?", "Rome."},
    {"What is the capital of Spain?", "Madrid."},
    {"What is the capital of Portugal?", "Lisbon."},
    {"What is the capital of Japan?", "Tokyo."},
    {"What is the capital of China?", "Beijing."},
    {"What is the capital of Canada?", "Ottawa."},
    {"What is the capital of Brazil?", "Brasilia."},
    {"What is the capital of Australia?", "Canberra."},
    {"What is the capital of Norway?", "Oslo."},
    {"What is the capital of Sweden?", "Stockholm."},
    {"What is the capital of Finland?", "Helsinki."},
    {"What is the capital of Poland?", "Warsaw."},
    {"What is the capital of Greece?", "Athens."},
    {"What is the capital of Austria?", "Vienna."},
    {"What is the capital of Belgium?", "Brussels."},
    {"What is the capital of Ireland?", "Dublin."},
    {"What is the capital of Denmark?", "Copenhagen."},
    {"What is the capital of Switzerland?", "Bern."},

    {"How many days are in a week?", "7."},
    {"How many months are in a year?", "12."},
    {"How many hours are in a day?", "24."},
    {"How many minutes are in an hour?", "60."},
    {"How many seconds are in a minute?", "60."},
    {"How many days are in a year?", "365."},
    {"How many hours are in two days?", "48."},
    {"How many minutes are in two hours?", "120."},
    {"How many seconds are in two minutes?", "120."},
    {"How many months are in two years?", "24."},

    {"How many sides does a triangle have?", "3."},
    {"How many sides does a square have?", "4."},
    {"How many sides does a pentagon have?", "5."},
    {"How many sides does a hexagon have?", "6."},
    {"How many sides does a heptagon have?", "7."},
    {"How many sides does an octagon have?", "8."},
    {"How many sides does a nonagon have?", "9."},
    {"How many sides does a decagon have?", "10."},
    {"How many degrees are in a right angle?", "90."},
    {"How many degrees are in a full circle?", "360."},

    {"What color is snow?", "White."},
    {"What color is grass?", "Green."},
    {"What color is the sky on a clear day?", "Blue."},
    {"What color is coal?", "Black."},
    {"What color is a ripe banana?", "Yellow."},
    {"What color are oranges usually?", "Orange."},
    {"What color are strawberries usually?", "Red."},
    {"What color is milk?", "White."},
    {"What color is a lemon?", "Yellow."},
    {"What color is a typical tomato?", "Red."},

    {"What planet do humans live on?", "Earth."},
    {"What is the largest planet in the Solar System?", "Jupiter."},
    {"What is the closest planet to the Sun?", "Mercury."},
    {"What star is at the center of our Solar System?", "The Sun."},
    {"How many planets are in the Solar System?", "8."},
    {"What is the natural satellite of Earth?", "The Moon."},
    {"What gas do humans need to breathe?", "Oxygen."},
    {"What gas do plants use during photosynthesis?", "Carbon dioxide."},
    {"What force pulls objects toward Earth?", "Gravity."},
    {"What is H2O commonly called?", "Water."},

    {"What is the opposite of hot?", "Cold."},
    {"What is the opposite of big?", "Small."},
    {"What is the opposite of fast?", "Slow."},
    {"What is the opposite of light?", "Dark."},
    {"What is the opposite of old?", "Young."},
    {"What is the opposite of high?", "Low."},
    {"What is the opposite of early?", "Late."},
    {"What is the opposite of open?", "Closed."},
    {"What is the opposite of full?", "Empty."},
    {"What is the opposite of strong?", "Weak."}
};

std::string MakeFact(std::mt19937& gen) {
    const Fact& fact = RandomElement(gen, FACTS);
    int variant = RandomInt(gen, 0, 3);

    if (variant == 0) {
        return "Question: " + fact.question 
            + " Answer: " + fact.answer + " <END>";
    }

    if (variant == 1) {
        return "Question: Please answer this: " + fact.question
            + " Answer: " + fact.answer + " <END>";
    }

    if (variant == 2) {
        return "Question: Can you answer this question? " + fact.question
            + " Answer: " + fact.answer + " <END>";
    }

    return "Question: Tell me the answer to this: " + fact.question
        + " Answer: " + fact.answer + " <END>";
}

struct Definition {
    std::string question;
    std::string answer;
};

const std::vector<Definition> DEFINITIONS = {

    {"What is water?", "A liquid."},
    {"What is the Sun?", "A star."},
    {"What is a dog?", "An animal."},
    {"What is a cat?", "An animal."},
    {"What is a rose?", "A flower."},
    {"What is a tree?", "A plant."},
    {"What is ice?", "Frozen water."},
    {"What is rain?", "Water from clouds."},
    {"What is snow?", "Frozen water."},
    {"What is a bird?", "An animal."},
    {"What is a fish?", "An animal."},
    {"What is a computer?", "A machine."},
    {"What is a car?", "A vehicle."},
    {"What is a bicycle?", "A vehicle."},
    {"What is an apple?", "A fruit."},
    {"What is a banana?", "A fruit."},
    {"What is a carrot?", "A vegetable."},
    {"What is the Moon?", "A natural satellite."},
    {"What is Mars?", "A planet."},
    {"What is Jupiter?", "A planet."},
    {"What is a whale?", "An animal."},
    {"What is a table?", "Furniture."},
    {"What is a chair?", "Furniture."},
    {"What is a plane?", "A vehicle."},
    {"What is a boat?", "A vehicle."},
    {"What is bread?", "Food."},
    {"What is cheese?", "Food."},
    {"What is coffee?", "A drink."},
    {"What is tea?", "A drink."}
};

std::string MakeDefinition(std::mt19937& gen) {
    const Definition& definition = RandomElement(gen, DEFINITIONS);
    int variant = RandomInt(gen, 0, 2);

    if (variant == 0) {
        return "Question: " + definition.question
            + " Answer: " + definition.answer + " <END>";
    }

    if (variant == 1) {
        return "Question: Please answer this: " + definition.question
            + " Answer: " + definition.answer + " <END>";
    }

    return "Question: Can you answer this question? " + definition.question
        + " Answer: " + definition.answer + " <END>";
}

struct Association {
    std::string question;
    std::string answer;
};

const std::vector<Association> ASSOCIATIONS = {

    {"What do bees produce?", "Honey."},
    {"What do cows produce?", "Milk."},
    {"What do chickens lay?", "Eggs."},
    {"What do birds use to fly?", "Wings."},
    {"What do fish use to swim?", "Fins."},
    {"What do humans use to see?", "Eyes."},
    {"What do humans use to hear?", "Ears."},
    {"What do humans use to smell?", "Nose."},
    {"What do humans use to walk?", "Legs."},
    {"What do humans use to think?", "The brain."},

    {"What do fish live in?", "Water."},
    {"Where do birds usually live?", "Nests."},
    {"Where do lions usually live?", "The wild."},
    {"Where do polar bears live?", "The Arctic."},
    {"Where do camels live?", "Deserts."},

    {"What falls from clouds?", "Rain."},
    {"What shines during the day?", "The Sun."},
    {"What shines at night?", "The Moon."},
    {"What melts when heated?", "Ice."},
    {"What freezes when cooled?", "Water."},

    {"What do we use to write?", "A pen."},
    {"What do we use to cut paper?", "Scissors."},
    {"What do we use to tell time?", "A clock."},
    {"What do we use to open a door?", "A key."},
    {"What do we use to call someone?", "A phone."},

    {"What fruit is usually yellow?", "A banana."},
    {"What fruit is usually red?", "An apple."},
    {"What vegetable is usually orange?", "A carrot."},
    {"What drink is made from coffee beans?", "Coffee."},
    {"What food is made from flour and baked?", "Bread."}
};

std::string MakeAssociation(std::mt19937& gen) {
    const Association& association = RandomElement(gen, ASSOCIATIONS);
    int variant = RandomInt(gen, 0, 2);

    if (variant == 0) {
        return "Question: " + association.question
            + " Answer: " + association.answer + " <END>";
    }

    if (variant == 1) {
        return "Question: Please answer this: " + association.question
            + " Answer: " + association.answer + " <END>";
    }

    return "Question: Can you answer this question? " + association.question
        + " Answer: " + association.answer + " <END>";
}

std::string MakeLogic(std::mt19937& gen) {
    int type = RandomInt(gen, 0, 3);
    int a = RandomInt(gen, 1, 100);
    int b = RandomInt(gen, 1, 100);

    if (type == 0) {
        return "Question: Is " + std::to_string(a) + " greater than "
            + std::to_string(b) + "? Answer: " 
            + std::string(a > b ? "Yes." : "No.") + " <END>";
    }

    if (type == 1) {
        return "Question: Is " + std::to_string(a) + " smaller than "
            + std::to_string(b) + "? Answer: "
            + std::string(a < b ? "Yes." : "No.") + " <END>";
    }

    if (type == 2) {
        return "Question: Is " + std::to_string(a) + " equal to "
            + std::to_string(b) + "? Answer: "
            + std::string(a == b ? "Yes." : "No.") + " <END>";
    }

    return "Question: Is " + std::to_string(a) + " not equal to "
        + std::to_string(b) + "? Answer: "
        + std::string(a != b ? "Yes." : "No.") + " <END>";
}

void WriteDataset(const std::string& path, 
        const std::vector<std::string>& dataset) {
    std::ofstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open: " + path);
    }

    for (const std::string& example : dataset) {
        file << example << '\n';
    }
}

void GenerateDataset() {
    std::mt19937 gen(TRAIN_SEED);
    std::vector<std::string> dataset;
    dataset.reserve(DATASET_SIZE);

    size_t fact_count = 0;
    size_t definition_count = 0;
    size_t association_count = 0;
    size_t logic_count = 0;

    const size_t FACT_COUNT = 7000;
    const size_t DEFINITION_COUNT = 5000;
    const size_t ASSOCIATION_COUNT = 4000;
    const size_t LOGIC_COUNT = 4000;

    for (size_t i = 0; i < FACT_COUNT; i++) {
        dataset.push_back(MakeFact(gen));
        ++fact_count;
    }

    for (size_t i = 0; i < DEFINITION_COUNT; i++) {
        dataset.push_back(MakeDefinition(gen));
        ++definition_count;
    }

    for (size_t i = 0; i < ASSOCIATION_COUNT; i++) {
        dataset.push_back(MakeAssociation(gen));
        ++association_count;
    }

    for (size_t i = 0; i < LOGIC_COUNT; i++) {
        dataset.push_back(MakeLogic(gen));
        ++logic_count;
    }

    if (dataset.size() != DATASET_SIZE) {
        throw std::runtime_error("Dataset size mismatch");
    }

    std::shuffle(dataset.begin(), dataset.end(), gen);
    WriteDataset(OUTPUT_PATH, dataset);

    std::cout << "Dataset generated successfully.\n\n"
        << "File:             " << OUTPUT_PATH << '\n'
        << "Total examples:   " << dataset.size() << '\n'
        << "Facts:            " << fact_count << '\n'
        << "Definitions:      " << definition_count << '\n'
        << "Associations:     " << association_count << '\n'
        << "Logic:            " << logic_count << '\n'
        << "Seed:              " << TRAIN_SEED << '\n';
}

int main() {

    std::cout << "========================================\n"
        << "ENGLISH DATASET GENERATOR\n"
        << "========================================\n\n";

    try {
        GenerateDataset();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    std::cout << "\nDone.\n";
    return 0;
}