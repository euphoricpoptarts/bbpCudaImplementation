#pragma once
#include <boost/heap/priority_queue.hpp>
#include <boost/asio/io_context.hpp>
#include <mutex>
#include <list>
#include "kernel.cuh"
#include <chrono>

class digitData;
class progressData;
struct segmentResult {
	digitData * digit;
	sJ result;
	double totalTime;
	std::chrono::high_resolution_clock::time_point timeValid;
};

struct segmentRequest {
	progressData * controller;
	std::list<std::string> controlledUuids;
	std::chrono::high_resolution_clock::time_point timeValid;
};

struct progressUpdate {
	digitData * digit;
	double progress;
	double timeElapsed;
	std::chrono::high_resolution_clock::time_point timeValid;
};

struct comparatorX {
	bool operator()(const segmentResult& a, const segmentResult& b) const
	{
		return a.timeValid < b.timeValid;
	}
	bool operator()(const segmentRequest& a, const segmentRequest& b) const
	{
		return a.timeValid < b.timeValid;
	}
	bool operator()(const progressUpdate& a, const progressUpdate& b) const
	{
		return a.timeValid < b.timeValid;
	}
};

class restClientDelegator
{
private:
	boost::heap::priority_queue<segmentResult, boost::heap::compare<comparatorX>> resultsToSave;
	boost::heap::priority_queue<segmentRequest, boost::heap::compare<comparatorX>> requestsForWork;
	boost::heap::priority_queue<progressUpdate, boost::heap::compare<comparatorX>> progressUpdates;
	std::mutex resultsMtx, requestsMtx, progressMtx;
	void processResultsQueue(boost::asio::io_context& ioc, std::chrono::high_resolution_clock::time_point validBefore);
	void processRequestsQueue(boost::asio::io_context& ioc, std::chrono::high_resolution_clock::time_point validBefore);
	void processProgressUpdatesQueue(boost::asio::io_context& ioc, std::chrono::high_resolution_clock::time_point validBefore);

public:
	void addResultToQueue(digitData * digit, sJ result, double totalTime);
	void addRequestToQueue(progressData * controller, std::list<std::string> controlledUuids);
	void addProgressUpdateToQueue(digitData * digit, double progress, double timeElapsed);
	void addResultToQueue(digitData * digit, sJ result, double totalTime, std::chrono::high_resolution_clock::time_point validAfter);
	void addRequestToQueue(progressData * controller, std::list<std::string> controlledUuids, std::chrono::high_resolution_clock::time_point validAfter);
	void monitorQueues();
};

