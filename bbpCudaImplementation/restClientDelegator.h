#pragma once
#include <boost/heap/priority_queue.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/beast/http.hpp>
#include <mutex>
#include <list>
#include "kernel.cuh"
#include <chrono>

class digitData;
class progressData;

struct apiCall {
	std::string endpoint, body;
	boost::beast::http::verb verb;
	std::function<void()> failHandle;
	std::function<void(boost::property_tree::ptree)> successHandle;
	std::chrono::steady_clock::time_point timeValid;

	bool operator()(const apiCall* a, const apiCall* b) const {
		return a->timeValid > b->timeValid;
	}
};

class restClientDelegator
{
private:
	boost::heap::priority_queue<apiCall*, boost::heap::compare<apiCall>> apiCallQueue;
	std::mutex queueMtx;
	void processQueue(boost::asio::io_context& ioc, const std::chrono::high_resolution_clock::time_point validBefore);
	void retryOnFail(apiCall * toRetry);
	static void noopFail(apiCall * failed);
	static void noopSuccess(apiCall * succeeded, const boost::property_tree::ptree pt);
	static void processRequest(progressData * data, std::list<std::string> controlledUuids, restClientDelegator * returnToSender, apiCall * call, const boost::property_tree::ptree& pt);

public:
	void addResultPutToQueue(digitData * digit, sJ result, double totalTime);
	void addWorkGetToQueue(progressData * controller, std::list<std::string> controlledUuids);
	void addProgressPutToQueue(digitData * digit, double progress, double timeElapsed);
	void monitorQueues();
};
