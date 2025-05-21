#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <cmath>
#include <random>

namespace ob = ompl::base;
namespace og = ompl::geometric;

// Assuming Polygon structure and isPointInObstacle function
struct Polygon {
    std::vector<std::pair<double, double>> vertices;
};

// Assuming this function checks if a given point is inside any of the provided polygons
bool isPointInObstacle(double x, double y, const std::vector<Polygon> &obstacles) {
    for (const auto &polygon : obstacles) {
        int n = polygon.vertices.size();
        bool inside = false;
        for (int i = 0, j = n - 1; i < n; j = i++) {
            auto [xi, yi] = polygon.vertices[i];
            auto [xj, yj] = polygon.vertices[j];
            if (((yi > y) != (yj > y)) &&
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
                inside = !inside;
            }
        }
        if (inside) return true;
    }
    return false;
}

// Assuming m_scene is a class or object that provides obstacle polygons
class Scene {
public:
    std::vector<Polygon> GetObstacles() const {
        return obstacles;
    }

    void LoadObstacles(const std::string &filename) {
        std::ifstream infile(filename);
        if (!infile) {
            std::cerr << "Could not open the polygon file!" << std::endl;
            exit(1);
        }

        int numPolygons;
        infile >> numPolygons;

        for (int i = 0; i < numPolygons; ++i) {
            int numVertices;
            infile >> numVertices;

            Polygon polygon;
            for (int j = 0; j < numVertices; ++j) {
                double x, y;
                infile >> x >> y;
                polygon.vertices.emplace_back(x, y);
            }
            obstacles.push_back(polygon);
        }
    }

    void SetBoundaries(double minX, double maxX, double minY, double maxY) {
        mazeMinX = minX;
        mazeMaxX = maxX;
        mazeMinY = minY;
        mazeMaxY = maxY;
    }

    double getMinX() const { return mazeMinX; }
    double getMaxX() const { return mazeMaxX; }
    double getMinY() const { return mazeMinY; }
    double getMaxY() const { return mazeMaxY; }

private:
    std::vector<Polygon> obstacles;
    double mazeMinX = -40; 
    double mazeMaxX = 40;
    double mazeMinY = -40;
    double mazeMaxY = 40;
};

Scene m_scene;  // Global instance of the scene

// Global path container to store the RRT path points
std::vector<std::pair<double, double>> rrtPath;

void planRRT(double radius) {
    double robotRadius = radius;
    int num_checks = 20;  // Number of points to check around the robot's radius

    // Define the SE2 state space (for 2D position + orientation)
    auto space = std::make_shared<ob::SE2StateSpace>();

    // Set bounds for the state space (adjust as necessary)
    ob::RealVectorBounds bounds(2);
    bounds.setLow(-100);  // Minimum bounds for x and y
    bounds.setHigh(100);  // Maximum bounds for x and y
    space->setBounds(bounds);

    // Initialize SimpleSetup
    auto ss = std::make_shared<og::SimpleSetup>(space);

    // Define the start and goal states
    ob::ScopedState<ob::SE2StateSpace> startState(space);
    startState->setX(-36.6031);  // Replace with your desired start x-coordinate
    startState->setY(-36.4332);  // Replace with your desired start y-coordinate
    startState->setYaw(0.0);  // Initial orientation

    ob::ScopedState<ob::SE2StateSpace> goalState(space);
    goalState->setX(36.0982);  // Replace with your desired goal x-coordinate
    goalState->setY(36.363);  // Replace with your desired goal y-coordinate
    goalState->setYaw(0.0);  // Final orientation

    // Set the start and goal states in the SimpleSetup
    ss->setStartAndGoalStates(startState, goalState);

    // Retrieve obstacles from the scene
    auto obstacles = m_scene.GetObstacles();

    // Define the state validity checker
    ss->setStateValidityChecker([&](const ob::State *state) {
        const auto *se2state = state->as<ob::SE2StateSpace::StateType>();
        double x = se2state->getX();
        double y = se2state->getY();

        // Check if the state is outside maze boundaries
        if (x - robotRadius < m_scene.getMinX() || x + robotRadius > m_scene.getMaxX() ||
            y - robotRadius < m_scene.getMinY() || y + robotRadius > m_scene.getMaxY()) {
            return false;
        }

        // Check multiple points around the robot's radius to ensure no collision
        for (int i = 0; i < num_checks; ++i) {
            double angle = 2 * M_PI * i / num_checks;
            double check_x = x + robotRadius * cos(angle);
            double check_y = y + robotRadius * sin(angle);

            if (isPointInObstacle(check_x, check_y, obstacles)) {
                return false;
            }
        }
        return true;
    });

    // Set state validity checking resolution (higher value = less precision, faster)
    ss->getSpaceInformation()->setStateValidityCheckingResolution(0.003);

    auto planner = std::make_shared<og::RRT>(ss->getSpaceInformation());
    ss->setPlanner(planner);

    ob::PlannerStatus solved = ss->solve(26.0);

    if (solved) {
        std::cout << "Found solution:\n";
        ss->simplifySolution();
        og::PathGeometric path = ss->getSolutionPath();
        path.printAsMatrix(std::cout);

        // Store the path points in rrtPath
        rrtPath.clear();
        for (size_t j = 0; j < path.getStateCount(); ++j) {
            const auto *state = path.getState(j)->as<ob::SE2StateSpace::StateType>();
            double x = state->getX();
            double y = state->getY();
            rrtPath.push_back({x, y});
        }
    } else {
        std::cout << "No solution found.\n";
    }
}

std::vector<std::pair<double, double>> prmPath;

void planPRM(double radius) {
    double robotRadius = radius;
    int num_checks = 20;  // Number of points to check around the robot's radius

    // Define the SE2 state space (for 2D position + orientation)
    auto space = std::make_shared<ob::SE2StateSpace>();

    // Set bounds for the state space (adjust as necessary)
    ob::RealVectorBounds bounds(2);
    bounds.setLow(-100);  // Minimum bounds for x and y
    bounds.setHigh(100);  // Maximum bounds for x and y
    space->setBounds(bounds);

    // Initialize SimpleSetup
    auto ss = std::make_shared<og::SimpleSetup>(space);

    // Define the start and goal states
    ob::ScopedState<ob::SE2StateSpace> startState(space);
    startState->setX(-36.6031);  // Replace with your desired start x-coordinate
    startState->setY(-36.4332);  // Replace with your desired start y-coordinate
    startState->setYaw(0.0);  // Initial orientation

    ob::ScopedState<ob::SE2StateSpace> goalState(space);
    goalState->setX(36.0982);  // Replace with your desired goal x-coordinate
    goalState->setY(36.363);  // Replace with your desired goal y-coordinate
    goalState->setYaw(0.0);  // Final orientation

    // Set the start and goal states in the SimpleSetup
    ss->setStartAndGoalStates(startState, goalState);

    // Retrieve obstacles from the scene
    auto obstacles = m_scene.GetObstacles();

    // Define the state validity checker
    ss->setStateValidityChecker([&](const ob::State *state) {
        const auto *se2state = state->as<ob::SE2StateSpace::StateType>();
        double x = se2state->getX();
        double y = se2state->getY();

        // Check if the state is outside maze boundaries
        if (x - robotRadius < m_scene.getMinX() || x + robotRadius > m_scene.getMaxX() ||
            y - robotRadius < m_scene.getMinY() || y + robotRadius > m_scene.getMaxY()) {
            return false;
        }

        // Check multiple points around the robot's radius to ensure no collision
        for (int i = 0; i < num_checks; ++i) {
            double angle = 2 * M_PI * i / num_checks;
            double check_x = x + robotRadius * cos(angle);
            double check_y = y + robotRadius * sin(angle);

            if (isPointInObstacle(check_x, check_y, obstacles)) {
                return false;
            }
        }
        return true;
    });

    // Set state validity checking resolution (higher value = less precision, faster)
    ss->getSpaceInformation()->setStateValidityCheckingResolution(0.003);

    auto planner = std::make_shared<og::PRM>(ss->getSpaceInformation());
    ss->setPlanner(planner);

    ob::PlannerStatus solved = ss->solve(26.0);

    if (solved) {
        std::cout << "Found solution:\n";
        ss->simplifySolution();
        og::PathGeometric path = ss->getSolutionPath();
        path.printAsMatrix(std::cout);

        prmPath.clear();
        for (size_t j = 0; j < path.getStateCount(); ++j) {
            const auto *state = path.getState(j)->as<ob::SE2StateSpace::StateType>();
            double x = state->getX();
            double y = state->getY();
            prmPath.push_back({x, y});
        }
    } else {
        std::cout << "No solution found.\n";
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_polygons_file> <RRT|PRM>" << std::endl;
        return 1;
    }

    std::string obsts_file = argv[1];
    std::string algorithm = argv[2];

    m_scene.LoadObstacles(obsts_file);

    m_scene.SetBoundaries(-40, 40, -40, 40);  

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.5, 1.5);
    double robotRadius = dis(gen);

    std::cout << "Robot radius: " << robotRadius << std::endl;

    if (algorithm == "RRT") {
        planRRT(robotRadius);
    } else if (algorithm == "PRM") {
        planPRM(robotRadius);
    } else {
        std::cerr << "Invalid algorithm specified. Use 'RRT' or 'PRM'." << std::endl;
        return 1;
    }

    std::string baseName(obsts_file); 

    size_t lastSlash = baseName.rfind('/'); 
    if (lastSlash != std::string::npos) {
        baseName = baseName.substr(lastSlash + 1);
    }

    std::ostringstream filePath; 
    if (algorithm == "RRT"){
        filePath << "paths-rrt/" << baseName; 
        std::ofstream outFile(filePath.str(), std::ios::app); 
        outFile << "@" << std::endl; 
        for (const auto &point : rrtPath){
            outFile << point.first << " " << point.second << std::endl;
        }
        outFile << "Robot Radius:" << " " << robotRadius << std::endl; 
    } else if (algorithm == "PRM") {
        filePath << "paths-prm/" << baseName; 
        std::ofstream outFile(filePath.str(), std::ios::app); 
        outFile << "@" << std::endl; 
        for (const auto &point : prmPath){
            outFile << point.first << " " << point.second << std::endl;
        }
        outFile << "Robot Radius:" << " " << robotRadius << std::endl; 

    }


    return 0;
}