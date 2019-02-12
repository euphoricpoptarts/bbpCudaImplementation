#pragma once

//modified from boost examples: https://www.boost.org/doc/libs/1_69_0/libs/beast/example/http/client/async/http_client_async.cpp

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <sstream>
#include <queue>
#include <list>
#include "digitData.hpp"
#include "kernel.cuh"
#include "progressData.h"

namespace ip = boost::asio::ip;       // from <boost/asio/ip/tcp.hpp>
namespace http = boost::beast::http;    // from <boost/beast/http.hpp>

										//------------------------------------------------------------------------------

										// Report a failure
void
fail(boost::system::error_code ec, char const* what)
{
	std::cerr << what << ": " << ec.message() << "\n";
}

struct segmentResult {
	digitData digit;
	sJ result;
	double totalTime;
};

struct segmentRequest {
	progressData * controller;
	std::list<std::string> controlledUuids;
};

// Performs an HTTP GET and prints the response
class session : public std::enable_shared_from_this<session>
{
	ip::tcp::resolver resolver_;
	ip::tcp::socket socket_;
	boost::beast::flat_buffer buffer_; // (Must persist between reads)
	http::request<http::empty_body> req_;
	http::response<http::string_body> res_;
	char const* host;
	char const* port;
	char const* target;
	int version;
	std::function<void (boost::property_tree::ptree)> processResponse;

public:
	// Resolver and socket require an io_context
	explicit
		session(boost::asio::io_context& ioc, char const* host,
			char const* port,
			char const* target,
			int version)
		: resolver_(ioc)
		, socket_(ioc)
		, host(host)
		, port(port)
		, target(target)
		, version(version)
	{
	}

	// Start the asynchronous operation
	void
		run(std::function<void (boost::property_tree::ptree)> processResponse)
	{
		this->processResponse = processResponse;
		// Set up an HTTP GET request message
		req_.version(version);
		req_.method(http::verb::get);
		req_.target(target);
		req_.set(http::field::host, host);
		req_.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);

		// Look up the domain name
		resolver_.async_resolve(
			host,
			port,
			std::bind(
				&session::on_resolve,
				shared_from_this(),
				std::placeholders::_1,
				std::placeholders::_2));
	}

	void
		on_resolve(
			boost::system::error_code ec,
			ip::tcp::resolver::results_type results)
	{
		if (ec)
			return fail(ec, "resolve");

		// Make the connection on the IP address we get from a lookup
		boost::asio::async_connect(
			socket_,
			results.begin(),
			results.end(),
			std::bind(
				&session::on_connect,
				shared_from_this(),
				std::placeholders::_1));
	}

	void
		on_connect(boost::system::error_code ec)
	{
		if (ec)
			return fail(ec, "connect");

		// Send the HTTP request to the remote host
		http::async_write(socket_, req_,
			std::bind(
				&session::on_write,
				shared_from_this(),
				std::placeholders::_1,
				std::placeholders::_2));
	}

	void
		on_write(
			boost::system::error_code ec,
			std::size_t bytes_transferred)
	{
		boost::ignore_unused(bytes_transferred);

		if (ec)
			return fail(ec, "write");

		// Receive the HTTP response
		http::async_read(socket_, buffer_, res_,
			std::bind(
				&session::on_read,
				shared_from_this(),
				std::placeholders::_1,
				std::placeholders::_2));
	}

	void
		on_read(
			boost::system::error_code ec,
			std::size_t bytes_transferred)
	{
		boost::ignore_unused(bytes_transferred);

		if (ec)
			return fail(ec, "read");

		boost::property_tree::ptree pt;

		std::stringstream ss(res_.body());
		boost::property_tree::read_json(ss, pt);

		processResponse(pt);

		// Gracefully close the socket
		socket_.shutdown(ip::tcp::socket::shutdown_both, ec);

		// not_connected happens sometimes so don't bother reporting it.
		if (ec && ec != boost::system::errc::not_connected)
			return fail(ec, "shutdown");

		// If we get here then the connection is closed gracefully
	}

	static void processRequest(progressData * data, boost::property_tree::ptree pt) {
		if (!pt.empty()) {
			std::cout << "response received" << std::endl;
			uint64 sumEnd = std::stoull(pt.get<std::string>("segmentEnd"));
			uint64 segmentBegin = std::stoull(pt.get<std::string>("segmentStart"));
			uint64 exponent = std::stoull(pt.get<std::string>("segmentEnd"));
			digitData * workUnit = new digitData(sumEnd, exponent, segmentBegin, 0);
			data->assignWork(workUnit);
		}
	}

	static void processResult(boost::property_tree::ptree pt) {
		if (!pt.empty()) {
			for (auto iter = pt.begin(); iter != pt.end(); iter++) {
				std::cout << iter->first << ":" << iter->second.get_value<std::string>() << std::endl;
			}
			std::cout << "Getting by value .get('segment'): " << pt.get<std::string>("segment") << std::endl;
		}
		else {
			std::cout << pt.get_value<std::string>() << std::endl;
		}
		std::cout << std::endl;
	}
};

class restClientDelegator {
private:
	std::list<segmentResult> resultsToSave;
	std::list<segmentRequest> requestsForWork;
	std::mutex resultsMtx, requestsMtx;

public:
	void addResultToQueue(digitData digit, sJ result, double totalTime) {
		resultsMtx.lock();
		resultsToSave.push_back(segmentResult{digit, result, totalTime});
		resultsMtx.unlock();
	}

	void addRequestToQueue(progressData * controller, std::list<std::string> controlledUuids) {
		requestsMtx.lock();
		requestsForWork.push_back(segmentRequest{ controller, controlledUuids });
		requestsMtx.unlock();
	}

	void monitorQueues() {
		while (!stop) {
			boost::asio::io_context ioc;
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
				std::cout << "SNEEDing request" << std::endl;
				std::function<void(boost::property_tree::ptree)> f = std::bind(&session::processRequest, request.controller, std::placeholders::_1);
				std::make_shared<session>(ioc, "127.0.0.1", "5000", "/getSegment", 11)->run(f);
			}
			requestsForWork.clear();
			requestsMtx.unlock();
			if(requestsMade) ioc.run();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}
};