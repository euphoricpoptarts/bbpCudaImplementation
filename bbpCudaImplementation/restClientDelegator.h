#pragma once
#include <mutex>
#include <list>
#include "kernel.cuh"
#include "digitData.hpp"
#include "progressData.hpp"
#include <boost/asio/connect.hpp>

struct segmentResult {
	digitData digit;
	sJ result;
	double totalTime;
};

struct segmentRequest {
	progressData * controller;
	std::list<std::string> controlledUuids;
};

class restClientDelegator {
private:
	std::list<segmentResult> resultsToSave;
	std::list<segmentRequest> requestsForWork;
	std::mutex resultsMtx, requestsMtx;

public:
	void addResultToQueue(digitData digit, sJ result, double totalTime) {
		resultsMtx.lock();
		resultsToSave.push_back(segmentResult{ digit, result, totalTime });
		resultsMtx.unlock();
	}

	void addRequestToQueue(progressData * controller, std::list<std::string> controlledUuids) {
		requestsMtx.lock();
		requestsForWork.push_back(segmentRequest{ controller, controlledUuids });
		requestsMtx.unlock();
	}

	void monitorQueues() {
		boost::asio::io_context ioc;
		while (!stop) {
			bool requestsMade = false;
			resultsMtx.lock();
			for (segmentResult& result : resultsToSave) {
				requestsMade = true;
				std::function<void(boost::property_tree::ptree)> f = std::bind(&session::processResult, std::placeholders::_1);
				std::make_shared<session>(ioc, "127.0.0.1", "5000", "/getSegment", 11)->run(f);
			}
			resultsToSave.clear();
			resultsMtx.unlock();
			requestsMtx.lock();
			for (segmentRequest& request : requestsForWork) {
				requestsMade = true;
				std::function<void(boost::property_tree::ptree)> f = std::bind(&session::processRequest, request.controller, std::placeholders::_1);
				std::make_shared<session>(ioc, "127.0.0.1", "5000", "/getSegment", 11)->run(f);
			}
			requestsForWork.clear();
			requestsMtx.unlock();
			if (requestsMade) ioc.run();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}
};