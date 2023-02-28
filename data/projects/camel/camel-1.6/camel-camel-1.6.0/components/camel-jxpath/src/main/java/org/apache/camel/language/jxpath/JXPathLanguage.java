/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.language.jxpath;

import org.apache.camel.Exchange;
import org.apache.camel.Expression;
import org.apache.camel.Predicate;
import org.apache.camel.spi.Language;

/**
 * <a href="http://commons.apache.org/jxpath/">JXPath</a> {@link Language}
 * provider
 */
public class JXPathLanguage implements Language {

    public Expression<Exchange> createExpression(String expression) {
        return new JXPathExpression(expression, Object.class);
    }

    public Predicate<Exchange> createPredicate(String predicate) {
        return new JXPathExpression(predicate, Boolean.class);
    }

}