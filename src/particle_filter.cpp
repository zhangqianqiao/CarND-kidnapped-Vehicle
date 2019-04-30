/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles
  
  // create normal distribution
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i=0; i < num_particles; i++) {
    Particle sample_p;
    sample_p.id = i;
    sample_p.x = dist_x(gen);
    sample_p.y = dist_y(gen);
    sample_p.theta = dist_theta(gen);
    sample_p.weight = 1.0;
    
    particles.push_back(sample_p);
    weights.push_back(1);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  float std_x = std_pos[0];
  float std_y = std_pos[1];
  float std_theta = std_pos[2];
  
  //create normal gaussian distribution for x, y and theta
  normal_distribution<double> normal_x(0.0, std_x);
  normal_distribution<double> normal_y(0.0, std_y);
  normal_distribution<double> normal_theta(0.0, std_theta);
  
  for (int i=0; i < num_particles; i++) {
    
    if (fabs(yaw_rate) < 0.0001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    particles[i].x += normal_x(gen);
    particles[i].y += normal_y(gen);
    particles[i].theta += normal_theta(gen);
  }
  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  double min_distance, distance ;
  int min_index;
  
  for (unsigned int i = 0; i < observations.size(); i++) {
    min_distance = 1000000.0;
    min_index = -1;
    for (unsigned int j = 0; j < predicted.size(); j++) {
      distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (min_distance > distance) {
        min_distance = distance;
        min_index = j;
      }
      observations[i].id = min_index;
    }
  }
}

double ParticleFilter::multivariateGaussian(const LandmarkObs &obs, const LandmarkObs &lm, double sigma[]) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sigma[0] * sigma[1]);
  // calculate exponent
  double exponent;
  exponent = (pow(obs.x - lm.x, 2) / (2 * pow(sigma[0], 2))) + (pow(obs.y - lm.y, 2) / (2 * pow(sigma[1], 2)));
 
  return gauss_norm * exp(-exponent);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  for (int p_id=0; p_id < num_particles; p_id++) {
    Particle p = particles[p_id];
    std::vector<LandmarkObs> predicted_obs;
    
    // Set landmarks with the range of the sensor
    for (unsigned int k=0; k < map_landmarks.landmark_list.size(); k++) {
      Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
      LandmarkObs lm_obs;
      double distance = dist(landmark.x_f, landmark.y_f, p.x, p.y);
      if (distance < sensor_range) {
        lm_obs.x = landmark.x_f;
        lm_obs.y = landmark.y_f;
        lm_obs.id = landmark.id_i;
        predicted_obs.push_back(lm_obs);
      }
    }
    // transform coordinates of all observations from current particle
    std::vector<LandmarkObs> transformed_obs;
    
    for (unsigned int obs_id = 0; obs_id < observations.size(); obs_id++) {
      LandmarkObs observation;
      observation.id = observations[obs_id].id;
		 observation.x = p.x + (observations[obs_id].x * cos(p.theta)) - (observations[obs_id].y * sin(p.theta));
		 observation.y = p.y + (observations[obs_id].x * sin(p.theta)) + (observations[obs_id].y * cos(p.theta));
      transformed_obs.push_back(observation);
    }
    //assign the observed measurement to predicted landmarks.
    dataAssociation(predicted_obs, transformed_obs);
    
    // update weights
    double prob = 1.0;

    for(unsigned int id = 0; id < transformed_obs.size(); id++) {
      LandmarkObs obs = transformed_obs[id];
      LandmarkObs lm = predicted_obs[obs.id];
      
      double pdf = multivariateGaussian(obs, lm, std_landmark);
      prob *= pdf;
    }
    particles[p_id].weight = prob;
    weights[p_id] = prob;
  }
 
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::discrete_distribution<int> dist_particles(weights.begin(), weights.end());
  std::vector<Particle> new_particles;
  
  new_particles.resize(num_particles);
  
  for (int i = 0; i < num_particles; i++) {
     new_particles[i] = particles[dist_particles(gen)];
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}