/*
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *   * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */
package org.apache.axis2.transport.nhttp;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Iterator;
import java.util.Map;

import javax.net.ssl.SSLContext;
import javax.xml.stream.XMLStreamException;

import org.apache.axiom.om.OMOutputFormat;
import org.apache.axis2.AxisFault;
import org.apache.axis2.Constants;
import org.apache.axis2.util.MessageContextBuilder;
import org.apache.axis2.util.Utils;
import org.apache.axis2.addressing.AddressingConstants;
import org.apache.axis2.addressing.EndpointReference;
import org.apache.axis2.context.ConfigurationContext;
import org.apache.axis2.context.MessageContext;
import org.apache.axis2.description.TransportOutDescription;
import org.apache.axis2.engine.AxisEngine;
import org.apache.axis2.engine.MessageReceiver;
import org.apache.axis2.handlers.AbstractHandler;
import org.apache.axis2.transport.OutTransportInfo;
import org.apache.axis2.transport.TransportSender;
import org.apache.axis2.transport.TransportUtils;
import org.apache.axis2.transport.MessageFormatter;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.http.HttpHost;
import org.apache.http.HttpResponse;
import org.apache.http.HttpStatus;
import org.apache.http.impl.nio.reactor.DefaultConnectingIOReactor;
import org.apache.http.impl.nio.reactor.SSLIOSessionHandler;
import org.apache.http.nio.NHttpClientConnection;
import org.apache.http.nio.NHttpClientHandler;
import org.apache.http.nio.reactor.ConnectingIOReactor;
import org.apache.http.nio.reactor.IOEventDispatch;
import org.apache.http.nio.reactor.SessionRequest;
import org.apache.http.nio.reactor.SessionRequestCallback;
import org.apache.http.params.BasicHttpParams;
import org.apache.http.params.HttpConnectionParams;
import org.apache.http.params.HttpParams;
import org.apache.http.params.HttpProtocolParams;
import org.apache.http.protocol.HTTP;
import org.apache.sandesha2.Sandesha2Constants;

/**
 * NIO transport sender for Axis2 based on HttpCore and NIO extensions
 */
public class HttpCoreNIOSender extends AbstractHandler implements TransportSender {

    private static final Log log = LogFactory.getLog(HttpCoreNIOSender.class);

    /** The Axis2 configuration context */
    private ConfigurationContext cfgCtx;
    /** The IOReactor */
    private ConnectingIOReactor ioReactor = null;
    /** The client handler */
    private NHttpClientHandler handler = null;
    /** The session request callback that calls back to the message receiver with errors */
    private final SessionRequestCallback sessionRequestCallback = getSessionRequestCallback();
    /** The SSL Context to be used */
    private SSLContext sslContext = null;
    /** The SSL session handler that manages hostname verification etc */
    private SSLIOSessionHandler sslIOSessionHandler = null;

    /**
     * Initialize the transport sender, and execute reactor in new seperate thread
     * @param cfgCtx the Axis2 configuration context
     * @param transportOut the description of the http/s transport from Axis2 configuration
     * @throws AxisFault thrown on an error
     */
    public void init(ConfigurationContext cfgCtx, TransportOutDescription transportOut) throws AxisFault {
        this.cfgCtx = cfgCtx;

        // is this an SSL Sender?
        sslContext = getSSLContext(transportOut);
        sslIOSessionHandler = getSSLIOSessionHandler(transportOut);

        // start the Sender in a new seperate thread
        Thread t = new Thread(new Runnable() {
            public void run() {
                executeClientEngine();
            }
        }, "HttpCoreNIOSender");
        t.start();
        log.info((sslContext == null ? "HTTP" : "HTTPS") + " Sender starting");
    }

    /**
     * Configure and start the IOReactor
     */
    private void executeClientEngine() {

        HttpParams params = getClientParameters();
        try {
            ioReactor = new DefaultConnectingIOReactor(
                NHttpConfiguration.getInstance().getClientIOWorkers(), params);
        } catch (IOException e) {
            log.error("Error starting the IOReactor", e);
        }

        handler = new ClientHandler(cfgCtx, params);
        IOEventDispatch ioEventDispatch = getEventDispatch(
            handler, sslContext, sslIOSessionHandler, params);

        try {
            ioReactor.execute(ioEventDispatch);
        } catch (InterruptedIOException ex) {
            log.fatal("Reactor Interrupted");
        } catch (IOException e) {
            log.fatal("Encountered an I/O error: " + e.getMessage(), e);
        }
        log.info("Sender Shutdown");
    }

    /**
     * Return the IOEventDispatch implementation to be used. This is overridden by the
     * SSL sender
     * @param handler
     * @param sslContext
     * @param params
     * @return
     */
    protected IOEventDispatch getEventDispatch(
        NHttpClientHandler handler, SSLContext sslContext,
        SSLIOSessionHandler sslIOSessionHandler, HttpParams params) {
        return new PlainClientIOEventDispatch(handler, params);
    }

    /**
     * Always return null, as this implementation does not support outgoing SSL
     * @param transportOut
     * @return null
     * @throws AxisFault
     */
    protected SSLContext getSSLContext(TransportOutDescription transportOut) throws AxisFault {
        return null;
    }

    /**
     * Create the SSL IO Session handler to be used by this listener
     * @param transportOut
     * @return always null
     */
    protected SSLIOSessionHandler getSSLIOSessionHandler(TransportOutDescription transportOut)
        throws AxisFault {
        return null;
    }

    /**
     * get HTTP protocol parameters to which the sender must adhere to
     * @return the applicable HTTP protocol parameters
     */
    private HttpParams getClientParameters() {
        NHttpConfiguration cfg = NHttpConfiguration.getInstance();
        HttpParams params = new BasicHttpParams();
        params
            .setIntParameter(HttpConnectionParams.SO_TIMEOUT,
                cfg.getProperty(HttpConnectionParams.SO_TIMEOUT, 60000))
            .setIntParameter(HttpConnectionParams.CONNECTION_TIMEOUT,
                cfg.getProperty(HttpConnectionParams.CONNECTION_TIMEOUT, 10000))
            .setIntParameter(HttpConnectionParams.SOCKET_BUFFER_SIZE,
                cfg.getProperty(HttpConnectionParams.SOCKET_BUFFER_SIZE, 8 * 1024))
            .setBooleanParameter(HttpConnectionParams.STALE_CONNECTION_CHECK,
                cfg.getProperty(HttpConnectionParams.STALE_CONNECTION_CHECK, 0) == 1)
            .setBooleanParameter(HttpConnectionParams.TCP_NODELAY,
                cfg.getProperty(HttpConnectionParams.TCP_NODELAY, 1) == 1)
            .setParameter(HttpProtocolParams.USER_AGENT, "Synapse-HttpComponents-NIO");
        return params;
    }

    /**
     * transport sender invocation from Axis2 core
     * @param msgContext message to be sent
     * @return the invocation response (always InvocationResponse.CONTINUE)
     * @throws AxisFault on error
     */
    public InvocationResponse invoke(MessageContext msgContext) throws AxisFault {

        // remove unwanted HTTP headers (if any from the current message)
        removeUnwantedHeaders(msgContext);

        EndpointReference epr = Util.getDestinationEPR(msgContext);
        if (epr != null) {
            if (!AddressingConstants.Final.WSA_NONE_URI.equals(epr.getAddress())) {
                sendAsyncRequest(epr, msgContext);
            } else {
                handleException("Cannot send message to " + AddressingConstants.Final.WSA_NONE_URI);
            }
        } else {
            if (msgContext.getProperty(Constants.OUT_TRANSPORT_INFO) != null) {
                if (msgContext.getProperty(Constants.OUT_TRANSPORT_INFO) instanceof ServerWorker) {
                    sendAsyncResponse(msgContext);
                } else {
                    sendUsingOutputStream(msgContext);
                }
            } else {
                handleException("No valid destination EPR or OutputStream to send message");
            }
        }

        if (msgContext.getOperationContext() != null) {
            msgContext.getOperationContext().setProperty(
                Constants.RESPONSE_WRITTEN, Constants.VALUE_TRUE);
        }

        return InvocationResponse.CONTINUE;
    }

    /**
     * Remove unwanted headers from the http response of outgoing request. These are headers which
     * should be dictated by the transport and not the user. We remove these as these may get
     * copied from the request messages
     * @param msgContext the Axis2 Message context from which these headers should be removed
     */
    private void removeUnwantedHeaders(MessageContext msgContext) {
        Map headers = (Map) msgContext.getProperty(MessageContext.TRANSPORT_HEADERS);
        if (headers != null && !headers.isEmpty()) {
            headers.remove(HTTP.CONN_DIRECTIVE);
            headers.remove(HTTP.TRANSFER_ENCODING);
            headers.remove(HTTP.DATE_DIRECTIVE);
            headers.remove(HTTP.SERVER_DIRECTIVE);
            headers.remove(HTTP.CONTENT_TYPE);
            headers.remove(HTTP.CONTENT_LEN);
            headers.remove(HTTP.USER_AGENT);
        }
    }

    /**
     * Send the request message asynchronously to the given EPR
     * @param epr the destination EPR for the message
     * @param msgContext the message being sent
     * @throws AxisFault on error
     */
    private void sendAsyncRequest(EndpointReference epr, MessageContext msgContext) throws AxisFault {
        try {
            URL url = new URL(epr.getAddress());
            int port = url.getPort();
            if (port == -1) {
                // use default
                if ("http".equals(url.getProtocol())) {
                    port = 80;
                } else if ("https".equals(url.getProtocol())) {
                    port = 443;
                }
            }
            HttpHost httpHost = new HttpHost(url.getHost(), port, url.getProtocol());

            Axis2HttpRequest axis2Req = new Axis2HttpRequest(epr, httpHost, msgContext);

            NHttpClientConnection conn = ConnectionPool.getConnection(url.getHost(), port);

            if (conn == null) {
                ioReactor.connect(new InetSocketAddress(url.getHost(), port),
                    null, axis2Req, sessionRequestCallback);
                log.debug("A new connection established");
            } else {
                ((ClientHandler) handler).submitRequest(conn, axis2Req);
                log.debug("An existing connection reused");
            }

            axis2Req.streamMessageContents();

        } catch (MalformedURLException e) {
            handleException("Malformed destination EPR : " + epr.getAddress(), e);
        } catch (IOException e) {
            handleException("IO Error while submiting request message for sending", e);
        }
    }

    /**
     * Send the passed in response message, asynchronously
     * @param msgContext the message context to be sent
     * @throws AxisFault on error
     */
    private void sendAsyncResponse(MessageContext msgContext) throws AxisFault {

        // remove unwanted HTTP headers (if any from the current message)
        removeUnwantedHeaders(msgContext);
        
        ServerWorker worker = (ServerWorker) msgContext.getProperty(Constants.OUT_TRANSPORT_INFO);
        HttpResponse response = worker.getResponse();

        OMOutputFormat format = Util.getOMOutputFormat(msgContext);
        MessageFormatter messageFormatter = TransportUtils.getMessageFormatter(msgContext);
        response.setHeader(
            HTTP.CONTENT_TYPE,
            messageFormatter.getContentType(msgContext, format, msgContext.getSoapAction()));

        // return http 500 when a SOAP fault is returned
        if (msgContext.getEnvelope().getBody().hasFault()) {
            response.setStatusCode(HttpStatus.SC_INTERNAL_SERVER_ERROR);
        }

        // if this is a dummy message to handle http 202 case with non-blocking IO
        // set the status code to 202 and the message body to an empty byte array (see below)
        if (Utils.isExplicitlyTrue(msgContext, NhttpConstants.SC_ACCEPTED) &&
                msgContext.getProperty(
                    Sandesha2Constants.MessageContextProperties.SEQUENCE_ID) == null) {
            response.setStatusCode(HttpStatus.SC_ACCEPTED);
        }

        // set any transport headers
        Map transportHeaders = (Map) msgContext.getProperty(MessageContext.TRANSPORT_HEADERS);
        if (transportHeaders != null && !transportHeaders.values().isEmpty()) {
            Iterator iter = transportHeaders.keySet().iterator();
            while (iter.hasNext()) {
                Object header = iter.next();
                Object value = transportHeaders.get(header);
                if (value != null && header instanceof String && value instanceof String) {
                    response.setHeader((String) header, (String) value);
                }
            }
        }
        worker.getServiceHandler().commitResponse(worker.getConn(), response);

        OutputStream out = worker.getOutputStream();
        try {
            if (Utils.isExplicitlyTrue(msgContext, NhttpConstants.SC_ACCEPTED) &&
                msgContext.getProperty(
                    Sandesha2Constants.MessageContextProperties.SEQUENCE_ID) == null) {
                // see comment above on the reasoning
                out.write(new byte[0]);
            } else {
                messageFormatter.writeTo(msgContext, format, out, true);
            }
            out.close();
        } catch (IOException e) {
            handleException("IO Error sending response message", e);
        }

        try {
            worker.getIs().close();
        } catch (IOException ignore) {}        
    }

    private void sendUsingOutputStream(MessageContext msgContext) throws AxisFault {
        OMOutputFormat format = Util.getOMOutputFormat(msgContext);
        MessageFormatter messageFormatter = TransportUtils.getMessageFormatter(msgContext);
        OutputStream out = (OutputStream) msgContext.getProperty(MessageContext.TRANSPORT_OUT);

        if (msgContext.isServerSide()) {
            OutTransportInfo transportInfo =
                (OutTransportInfo) msgContext.getProperty(Constants.OUT_TRANSPORT_INFO);

            if (transportInfo != null) {
                transportInfo.setContentType(
                messageFormatter.getContentType(msgContext, format, msgContext.getSoapAction()));
            } else {
                throw new AxisFault(Constants.OUT_TRANSPORT_INFO + " has not been set");
            }
        }

        try {
            messageFormatter.writeTo(msgContext, format, out, true);
            out.close();
        } catch (IOException e) {
            handleException("IO Error sending response message", e);
        }
    }


    public void cleanup(MessageContext msgContext) throws AxisFault {
        // do nothing
    }

    public void stop() {
        try {
            ioReactor.shutdown();
            log.info("Sender shut down");
        } catch (IOException e) {
            log.warn("Error shutting down IOReactor", e);
        }
    }

    /**
     * Return a SessionRequestCallback which gets notified of a connection failure
     * or an error during a send operation. This method finds the corresponding
     * Axis2 message context for the outgoing request, and find the message receiver
     * and sends a fault message back to the message receiver that is marked as
     * related to the outgoing request
     * @return a Session request callback
     */
    private static SessionRequestCallback getSessionRequestCallback() {
        return new SessionRequestCallback() {
            public void completed(SessionRequest request) {
            }

            public void failed(SessionRequest request) {
                handleError(request);
            }

            public void timeout(SessionRequest request) {
                handleError(request);
            }

            public void cancelled(SessionRequest sessionRequest) {

            }

            private void handleError(SessionRequest request) {
                if (request.getAttachment() != null &&
                    request.getAttachment() instanceof Axis2HttpRequest) {

                    Axis2HttpRequest axis2Request = (Axis2HttpRequest) request.getAttachment();
                    MessageContext mc = axis2Request.getMsgContext();
                    MessageReceiver mr = mc.getAxisOperation().getMessageReceiver();

                    try {
                        // this fault is NOT caused by the endpoint while processing. so we have to
                        // inform that this is a sending error (e.g. endpoint failure) and handle it
                        // differently at the message receiver.

                        Exception exception = request.getException();
                        MessageContext nioFaultMessageContext =
                            MessageContextBuilder.createFaultMessageContext(
                                /** this is not a mistake I do NOT want getMessage()*/
                                mc, new AxisFault(exception.toString(), exception));
                        nioFaultMessageContext.setProperty(NhttpConstants.SENDING_FAULT, Boolean.TRUE);
                        mr.receive(nioFaultMessageContext);
                        
                    } catch (AxisFault af) {
                        log.error("Unable to report back failure to the message receiver", af);
                    }
                }
            }
        };
    }

    // -------------- utility methods -------------
    private void handleException(String msg, Exception e) throws AxisFault {
        log.error(msg, e);
        throw new AxisFault(msg, e);
    }

    private void handleException(String msg) throws AxisFault {
        log.error(msg);
        throw new AxisFault(msg);
    }
}
